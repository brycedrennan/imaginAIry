import logging
import os
from functools import lru_cache

import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from PIL import Image
from torchvision.transforms.functional import normalize

from imaginairy.utils import get_cached_url_path
from imaginairy.vendored.codeformer.codeformer_arch import CodeFormer
from transformers import cached_path

logger = logging.getLogger(__name__)


class FaceRestorationCodeformer():
    def __init__(self):
        self.net = None
        self.face_helper = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self):
        if self.net is not None and self.face_helper is not None:
            self.net.to(self.device)
            return self.net, self.face_helper
        net = CodeFormer(dim_embd=512,
                         codebook_size=1024,
                         n_head=8,
                         n_layers=9,
                         connect_list=['32', '64', '128', '256']).to(self.device)

        ckpt_path = cached_path(os.getenv("CODEFORME_MODEL_PATH", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"))
        checkpoint = torch.load(ckpt_path)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
                                        save_ext='png', use_parse=True, device=self.device)
        self.net = net
        self.net.to(self.device)
        return net, self.face_helper

    def enhance_faces(self, img, fidelity=0):
        image = img.convert("RGB")
        np_image = np.array(image, dtype=np.uint8)
        np_image = np_image[:, :, ::-1]

        original_resolution = np_image.shape[0:2]

        self.create_model()
        self.face_helper.clean_all()
        self.face_helper.read_image(np_image)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=fidelity, adain=True)[0]  # noqa
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                logger.error(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)

        restored_img = self.face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        res = Image.fromarray(restored_img[:, :, ::-1])
        return res

        # if original_resolution != restored_img.shape[0:2]:
        #     restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1] / restored_img.shape[1],
        #                               fy=original_resolution[0] / restored_img.shape[0], interpolation=cv2.INTER_LINEAR)
        #
        # if shared.opts.face_restoration_unload:
        #     self.net.to(devices.cpu)
        #
        # return restored_img
#
# @lru_cache()
# def codeformer_model():
#
#
#
#     model = CodeFormer(
#         dim_embd=512,
#         codebook_size=1024,
#         n_head=8,
#         n_layers=9,
#         connect_list=["32", "64", "128", "256"],
#     ).to("cpu")
#     url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
#     ckpt_path = get_cached_url_path(url)
#     checkpoint = torch.load(ckpt_path)["params_ema"]
#     model.load_state_dict(checkpoint)
#     model.eval()
#     return model
#
#
# def enhance_faces(img, fidelity=0):
#     net = codeformer_model()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     face_helper = FaceRestoreHelper(
#         1,
#         face_size=512,
#         crop_ratio=(1, 1),
#         det_model="retinaface_resnet50",
#         save_ext="png",
#         use_parse=True,
#         device=device,
#     )
#     face_helper.clean_all()
#
#     image = img.convert("RGB")
#     np_img = np.array(image, dtype=np.uint8)
#     # rotate to BGR
#     np_img = np_img[:, :, ::-1]
#
#     face_helper.read_image(np_img)
#     # get face landmarks for each face
#     num_det_faces = face_helper.get_face_landmarks_5(
#         only_center_face=False, resize=640, eye_dist_threshold=5
#     )
#     logger.info(f"    Enhancing {num_det_faces} faces")
#     # align and warp each face
#     face_helper.align_warp_face()
#
#     # face restoration for each cropped face
#     for cropped_face in face_helper.cropped_faces:
#         # prepare data
#         cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
#         normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
#         cropped_face_t = cropped_face_t.unsqueeze(0).to("cpu")
#
#         try:
#             with torch.no_grad():
#                 output = net(cropped_face_t, w=fidelity, adain=True)[0]  # noqa
#                 restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
#             del output
#             torch.cuda.empty_cache()
#         except Exception as error:  # noqa
#             logger.error(f"\tFailed inference for CodeFormer: {error}")
#             restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
#
#         restored_face = restored_face.astype("uint8")
#         face_helper.add_restored_face(restored_face)
#
#     face_helper.get_inverse_affine(None)
#     # paste each restored face to the input image
#     restored_img = face_helper.paste_faces_to_input_image()
#     res = Image.fromarray(restored_img[:, :, ::-1])
#     return res
