from functools import lru_cache

import cv2
import numpy as np
import torch

from imaginairy.model_manager import get_cached_url_path
from imaginairy.utils import get_device


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netScoreOne = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, img_t):
        img_t = (img_t + 1) * 127.5
        img_t = img_t - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=img_t.dtype,
            device=img_t.device,
        ).view(1, 3, 1, 1)

        ten_vgg_one = self.netVggOne(img_t)
        ten_vgg_two = self.netVggTwo(ten_vgg_one)
        ten_vgg_thr = self.netVggThr(ten_vgg_two)
        ten_vgg_fou = self.netVggFou(ten_vgg_thr)
        ten_vgg_fiv = self.netVggFiv(ten_vgg_fou)

        ten_score_one = self.netScoreOne(ten_vgg_one)
        ten_score_two = self.netScoreTwo(ten_vgg_two)
        ten_score_thr = self.netScoreThr(ten_vgg_thr)
        ten_score_fou = self.netScoreFou(ten_vgg_fou)
        ten_score_fiv = self.netScoreFiv(ten_vgg_fiv)

        ten_score_one = torch.nn.functional.interpolate(
            input=ten_score_one,
            size=(img_t.shape[2], img_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ten_score_two = torch.nn.functional.interpolate(
            input=ten_score_two,
            size=(img_t.shape[2], img_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ten_score_thr = torch.nn.functional.interpolate(
            input=ten_score_thr,
            size=(img_t.shape[2], img_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ten_score_fou = torch.nn.functional.interpolate(
            input=ten_score_fou,
            size=(img_t.shape[2], img_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ten_score_fiv = torch.nn.functional.interpolate(
            input=ten_score_fiv,
            size=(img_t.shape[2], img_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return self.netCombine(
            torch.cat(
                [
                    ten_score_one,
                    ten_score_two,
                    ten_score_thr,
                    ten_score_fou,
                    ten_score_fiv,
                ],
                1,
            )
        )


@lru_cache(maxsize=1)
def hed_model():
    model = Network().to(get_device()).eval()
    model_path = get_cached_url_path(
        "https://huggingface.co/lllyasviel/ControlNet/resolve/38a62cbf79862c1bac73405ec8dc46133aee3e36/annotator/ckpts/network-bsds500.pth"
    )
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("module", "net"): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def create_hed_map(img_t):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hed_model().to(device)
    img_t = img_t.to(device)
    with torch.no_grad():
        edge = model(img_t)[0]
    return edge[0]


def nms(x, t, s):
    """make scribbles."""
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
