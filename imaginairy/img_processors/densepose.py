# adapted from https://github.com/Mikubill/sd-webui-controlnet/blob/0b90426254debf78bfc09d88c064d2caf0935282/annotator/densepose/densepose.py
import logging
import math
from enum import IntEnum
from functools import lru_cache
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from imaginairy import config
from imaginairy.utils.downloads import get_cached_url_path

logger = logging.getLogger(__name__)

N_PART_LABELS = 24


_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]
IntTupleBox = Tuple[int, int, int, int]


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def resize_image_with_pad_torch(
    img, resolution, upscale_method="bicubic", mode="constant"
):
    B, C, H_raw, W_raw = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(math.ceil(float(H_raw) * k))
    W_target = int(math.ceil(float(W_raw) * k))

    if k > 1:
        img = F.interpolate(
            img,
            size=(H_target, W_target),
            mode=upscale_method,
            align_corners=False,
        )
    else:
        img = F.interpolate(img, size=(H_target, W_target), mode="area")

    H_pad, W_pad = pad64(H_target), pad64(W_target)
    # print(f"image after resize but before padding: {img.shape}")
    img_padded = F.pad(img, (0, W_pad, 0, H_pad), mode=mode)

    def remove_pad(x):
        # print(
        #     f"remove_pad: x.shape: {x.shape}. H_target: {H_target}, W_target: {W_target}"
        # )
        return safer_memory(x[:H_target, :W_target, ...])

    return img_padded, remove_pad


def HWC3(x: np.ndarray) -> np.ndarray:
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    raise RuntimeError("unreachable")


@lru_cache(maxsize=1)
def get_densepose_model(
    filename="densepose_r101_fpn_dl.torchscript", base_url=config.DENSEPOSE_REPO_URL
):
    import torchvision  # noqa

    url = f"{base_url}/{filename}"
    torchscript_model_path = get_cached_url_path(url)
    logger.info(f"Loading densepose model {url} from {torchscript_model_path}")
    densepose = torch.jit.load(torchscript_model_path, map_location="cpu")
    return densepose


@lru_cache(maxsize=1)
def get_segment_result_visualizer():
    return DensePoseMaskedColormapResultsVisualizer(
        alpha=1,
        data_extractor=_extract_i_from_iuvarr,
        segm_extractor=_extract_i_from_iuvarr,
        val_scale=255.0 / N_PART_LABELS,
    )


def mask_to_bbox(mask_img_t):
    m = mask_img_t.nonzero()
    if m.numel() == 0:
        return None
    y0 = torch.min(m[:, 0])
    y1 = torch.max(m[:, 0])
    x0 = torch.min(m[:, 1])
    x1 = torch.max(m[:, 1])
    return x0, y0, x1, y1


def pad_bbox(bbox, max_height, max_width, pad=1):
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(max_width, x1 + pad)
    y1 = min(max_height, y1 + pad)
    return x0, y0, x1, y1


def square_bbox(bbox, max_height, max_width):
    """
    Adjusts the bounding box to make it as close to a square as possible while
    ensuring it does not exceed the max_size of the image and still includes
    the original bounding box contents.

    Args:
    - bbox: A tuple of (x0, y0, x1, y1) for the original bounding box.
    - max_size: A tuple of (max_width, max_height) representing the image size.

    Returns:
    - A tuple of (x0, y0, x1, y1) for the adjusted bounding box.
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    # Determine how much to adjust to make the bounding box square
    if width > height:
        diff = width - height
        half_diff = diff // 2
        y0 = max(0, y0 - half_diff)
        y1 = min(max_height, y1 + half_diff + (diff % 2))  # Add 1 if diff is odd
    elif height > width:
        diff = height - width
        half_diff = diff // 2
        x0 = max(0, x0 - half_diff)
        x1 = min(max_width, x1 + half_diff + (diff % 2))  # Add 1 if diff is odd

    # Ensure the bounding box is within the image boundaries
    x0 = max(0, min(x0, max_width - 1))
    y0 = max(0, min(y0, max_height - 1))
    x1 = max(0, min(x1, max_width))
    y1 = max(0, min(y1, max_height))

    return x0, y0, x1, y1


def _np_to_t(img_np):
    img_t = torch.from_numpy(img_np) / 255.0
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t.unsqueeze(0)
    return img_t


def generate_densepose_image(
    img: torch.Tensor,
    detect_resolution=512,
    upscale_method="bicubic",
    cmap="viridis",
    double_pass=False,
):
    assert_tensor_float_11_bchw(img)
    input_h, input_w = img.shape[-2:]
    if double_pass:
        first_densepose_img_np = _generate_densepose_image(
            img, detect_resolution, upscale_method, cmap, adapt_viridis_bg=False
        )
        first_densepose_img_t = _np_to_t(first_densepose_img_np)
        # convert the densepose image into a mask (every color other than black is part of the mask)
        densepose_img_mask = first_densepose_img_t[0].sum(dim=0) > 0
        # print(f"Mask shape: {densepose_img_mask.shape}")
        # bbox = masks_to_boxes(densepose_img_mask.unsqueeze(0)).to(torch.uint8)
        # crop image by bbox
        bbox = mask_to_bbox(densepose_img_mask)
        # print(f"bbox: {bbox}")

        if bbox is None:
            densepose_np = first_densepose_img_np
        else:
            bbox = pad_bbox(bbox, max_height=input_h, max_width=input_w, pad=10)
            # print(f"padded bbox: {bbox}")
            bbox = square_bbox(bbox, max_height=input_h, max_width=input_w)
            # print(f"boxed bbox: {bbox}")
            x0, y0, x1, y1 = bbox

            cropped_img = img[:, :, y0:y1, x0:x1]
            # print(f"cropped_img shape: {cropped_img.shape}")

            densepose_np = _generate_densepose_image(
                cropped_img,
                detect_resolution,
                upscale_method,
                cmap,
                adapt_viridis_bg=False,
            )
            # print(f"cropped densepose_np shape: {densepose_np.shape}")
            # print(
            #     f"pasting into first_densepose_img_np shape: {first_densepose_img_np.shape} at {y0}:{y1}, {x0}:{x1}"
            # )
            # paste denspose_np back into first_densepose_img_np using bbox
            first_densepose_img_np[y0:y1, x0:x1] = densepose_np
            densepose_np = first_densepose_img_np
    else:
        densepose_np = _generate_densepose_image(
            img, detect_resolution, upscale_method, cmap, adapt_viridis_bg=False
        )

    if cmap == "viridis":
        densepose_np[:, :, 0][densepose_np[:, :, 0] == 0] = 68
        densepose_np[:, :, 1][densepose_np[:, :, 1] == 0] = 1
        densepose_np[:, :, 2][densepose_np[:, :, 2] == 0] = 84

    return densepose_np


def _generate_densepose_image(
    img: torch.Tensor,
    detect_resolution=512,
    upscale_method="bicubic",
    cmap="viridis",
    adapt_viridis_bg=True,
) -> np.ndarray:
    assert_tensor_float_11_bchw(img)
    input_h, input_w = img.shape[-2:]
    # print(f"input_h: {input_h}, input_w: {input_w}")
    img, remove_pad = resize_image_with_pad_torch(
        img, detect_resolution, upscale_method
    )
    img = ((img + 1.0) * 127.5).to(torch.uint8)
    assert_tensor_uint8_255_bchw(img)
    H, W = img.shape[-2:]
    # print(f"reduced input img size (with padding): h{H}xw{W}")
    hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
    hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])
    densepose_model = get_densepose_model()
    pred_boxes, coarse_seg, fine_segm, u, v = densepose_model(img.squeeze(0))
    densepose_results = list(
        map(
            densepose_chart_predictor_output_to_result,
            pred_boxes,
            coarse_seg,
            fine_segm,
            u,
            v,
        )
    )
    cmaps = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "parula": cv2.COLORMAP_PARULA,
        "jet": cv2.COLORMAP_JET,
    }
    cv2_cmap = cmaps.get(cmap, cv2.COLORMAP_PARULA)
    result_visualizer = get_segment_result_visualizer()
    result_visualizer.mask_visualizer.cmap = cv2_cmap
    hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
    hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

    if cv2_cmap == cv2.COLORMAP_VIRIDIS and adapt_viridis_bg:
        hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
        hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
        hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
    # print(f"hint_image shape: {hint_image.shape}")
    detected_map = remove_pad(HWC3(hint_image))
    # print(f"detected_map shape (padding removed): {detected_map.shape}")
    # print(f"Resizing detected_map to original size: {input_w}x{input_h}")
    # if map is smaller than input size, scale it up
    if detected_map.shape[0] < input_h or detected_map.shape[1] < input_w:
        detected_map = cv2.resize(
            detected_map, (input_w, input_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        # scale it down
        detected_map = cv2.resize(
            detected_map, (input_w, input_h), interpolation=cv2.INTER_AREA
        )
    # print(f"detected_map shape (resized to original): {detected_map.shape}")
    return detected_map


def assert_ndarray_uint8_255_hwc(img):
    # assert input_image is ndarray with colors 0-255
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.max() <= 255
    assert img.min() >= 0


def assert_tensor_uint8_255_bchw(img):
    # assert input_image is a PyTorch tensor with colors 0-255 and dimensions (C, H, W)
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.uint8
    assert img.ndim == 4
    assert img.shape[1] == 3
    assert img.max() <= 255
    assert img.min() >= 0


def assert_tensor_float_11_bchw(img):
    # assert input_image is a PyTorch tensor with colors -1 to 1 and dimensions (C, H, W)
    if not isinstance(img, torch.Tensor):
        msg = f"Input image must be a PyTorch tensor, but got {type(img)}"
        raise TypeError(msg)

    if img.dtype not in (torch.float32, torch.float64, torch.float16):
        msg = f"Input image must be a float tensor, but got {img.dtype}"
        raise ValueError(msg)

    if img.ndim != 4:
        msg = f"Input image must be 4D (B, C, H, W), but got {img.ndim}D"
        raise ValueError(msg)

    if img.shape[1] != 3:
        msg = f"Input image must have 3 channels, but got {img.shape[1]}"
        raise ValueError(msg)
    if img.max() > 1 or img.min() < -1:
        msg = f"Input image must have values in [-1, 1], but got {img.min()} .. {img.max()}"
        raise ValueError(msg)


class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(
        box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode"
    ) -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            arr = torch.from_numpy(np.asarray(box)).clone() if is_numpy else box.clone()  # type: ignore

        assert to_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"
        assert from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)  # type: ignore
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                msg = f"Conversion from BoxMode {from_mode} to {to_mode} is not supported yet"
                raise NotImplementedError(msg)

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class MatrixVisualizer:
    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr: np.ndarray, mask: np.ndarray, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        image_target_bgr = image_bgr if self.inplace else image_bgr * 0

        x, y, w, h = (int(v) for v in bbox_xywh)
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        _EPSILON = 1e-6
        if np.any(matrix_scaled > 255 + _EPSILON):
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Matrix has values > {255 + _EPSILON} after "
                f"scaling, clipping to [0..255]"
            )
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha)
            + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(np.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == np.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == np.uint8


class DensePoseMaskedColormapResultsVisualizer:
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def visualize(
        self,
        image_bgr: np.ndarray,
        results,
    ) -> np.ndarray:
        for result in results:
            boxes_xywh, labels, uv = result
            iuv_array = torch.cat((labels[None].type(torch.float32), uv * 255.0)).type(
                torch.uint8
            )
            self.visualize_iuv_arr(image_bgr, iuv_array.cpu().numpy(), boxes_xywh)
        return image_bgr

    def visualize_iuv_arr(self, image_bgr, iuv_arr: np.ndarray, bbox_xywh) -> None:
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = (segm > 0).astype(np.uint8)
        self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


def make_int_box(box: torch.Tensor) -> IntTupleBox:
    int_box = [0, 0, 0, 0]
    int_box[0], int_box[1], int_box[2], int_box[3] = tuple(box.long().tolist())
    return int_box[0], int_box[1], int_box[2], int_box[3]


def densepose_chart_predictor_output_to_result(
    boxes: torch.Tensor, coarse_segm: torch.Tensor, fine_segm, u, v
):
    boxes = boxes.unsqueeze(0)
    coarse_segm = coarse_segm.unsqueeze(0)
    fine_segm = fine_segm.unsqueeze(0)
    u = u.unsqueeze(0)
    v = v.unsqueeze(0)
    boxes_xyxy_abs = boxes.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])  # type: ignore

    labels = resample_fine_and_coarse_segm_tensors_to_bbox(
        fine_segm, coarse_segm, box_xywh
    ).squeeze(0)
    uv = resample_uv_tensors_to_bbox(u, v, labels, box_xywh)
    return box_xywh, labels, uv


def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs: IntTupleBox
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm,
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(
            dim=1
        )
        * (coarse_segm_bbox > 0).long()
    )
    return labels


def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv
