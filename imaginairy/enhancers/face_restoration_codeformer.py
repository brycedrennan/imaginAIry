"""Code for enhancing facial images"""

import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize

from imaginairy.utils.downloads import get_cached_url_path
from imaginairy.vendored.basicsr.img_util import img2tensor, tensor2img
from imaginairy.vendored.codeformer.codeformer_arch import CodeFormer
from imaginairy.vendored.facexlib.utils.face_restoration_helper import FaceRestoreHelper

logger = logging.getLogger(__name__)


face_restore_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half_mode = face_restore_device == "cuda"


def load_file_from_url(
    url, model_dir=None, progress=True, file_name=None, save_dir=None
):
    return get_cached_url_path(url, category="facexlib")


@lru_cache(maxsize=1)
def patch_download_function_in_facexlib_modules():
    """Replaces the custom weights downloaded with the standard imaginairy one."""
    import imaginairy.vendored.facexlib.utils.misc
    from imaginairy.vendored.facexlib import (
        alignment,
        assessment,
        detection,
        headpose,
        matting,
        parsing,
        recognition,
        tracking,
        visualization,
    )

    modules = [
        alignment,
        assessment,
        detection,
        headpose,
        matting,
        parsing,
        recognition,
        tracking,
        visualization,
        imaginairy.vendored.facexlib.utils.misc,
    ]
    for m in modules:
        m.load_file_from_url = load_file_from_url


patch_download_function_in_facexlib_modules()


@lru_cache
def codeformer_model():
    model = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(face_restore_device)
    url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    ckpt_path = get_cached_url_path(url)
    checkpoint = torch.load(ckpt_path)["params_ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    if half_mode:
        model = model.half()
    return model


@lru_cache
def face_restore_helper():
    """
    Provide a singleton of FaceRestoreHelper.

    FaceRestoreHelper loads a model internally so we need to cache it
    or we end up with a memory leak
    """
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=face_restore_device,
    )
    return face_helper


def enhance_faces(img, fidelity=0):
    net = codeformer_model()

    face_helper = face_restore_helper()
    face_helper.clean_all()

    image = img.convert("RGB")
    np_img = np.array(image, dtype=np.uint8)
    # rotate to BGR
    np_img = np_img[:, :, ::-1]

    face_helper.read_image(np_img)
    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    logger.debug(f"Enhancing {num_det_faces} faces")
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for face_box, cropped_face in zip(face_helper.det_faces, face_helper.cropped_faces):
        x1, y1, x2, y2, scaling = face_box
        face_width = x2 - x1
        face_height = y2 - y1
        logger.debug(f"Face detected. size: {face_width:1f}x{face_height:.1f}")
        if face_width > 512 or face_height > 512:
            logger.debug(
                f"Face too large: ({face_width:.1f}x{face_height:.1f}). skipping enhancement"
            )
            face_helper.add_restored_face(cropped_face)
            continue

        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(face_restore_device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            logger.exception(f"\tFailed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    restored_img = face_helper.paste_faces_to_input_image()
    res = Image.fromarray(restored_img[:, :, ::-1])
    face_helper.clean_all()
    return res
