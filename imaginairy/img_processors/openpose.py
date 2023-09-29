import math
from collections import OrderedDict
from functools import lru_cache

import cv2
import matplotlib as mpl
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from torch import nn

from imaginairy.img_utils import torch_image_to_openvcv_img
from imaginairy.model_manager import get_cached_url_path
from imaginairy.utils import get_device


def pad_right_down_corner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    # transfer caffe model to pytorch which will match the layer name
    transfered_model_weights = {}
    for weights_name in model.state_dict():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights


# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


# image drawed by opencv is not good.
def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    mpl.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
            if show_number:
                cv2.putText(
                    canvas,
                    str(i),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    lineType=cv2.LINE_AA,
                )
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]
            ]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image

            x = max(x, 0)
            y = max(y, 0)

            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    # return value: [[x, y, w, True if left hand else False]].
    # width=height since the network require squared input.
    # x, y is the coordinate of top left
    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4],
            )
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class bodypose_model(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers have no relu layer
        no_relu_layers = [
            "conv5_5_CPM_L1",
            "conv5_5_CPM_L2",
            "Mconv7_stage2_L1",
            "Mconv7_stage2_L2",
            "Mconv7_stage3_L1",
            "Mconv7_stage3_L2",
            "Mconv7_stage4_L1",
            "Mconv7_stage4_L2",
            "Mconv7_stage5_L1",
            "Mconv7_stage5_L2",
            "Mconv7_stage6_L1",
            "Mconv7_stage6_L1",
        ]
        blocks = {}
        block0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3_CPM", [512, 256, 3, 1, 1]),
                ("conv4_4_CPM", [256, 128, 3, 1, 1]),
            ]
        )

        # Stage 1
        block1_1 = OrderedDict(
            [
                ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
            ]
        )

        block1_2 = OrderedDict(
            [
                ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
            ]
        )
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks[f"block{i}_1"] = OrderedDict(
                [
                    (f"Mconv1_stage{i}_L1", [185, 128, 7, 1, 3]),
                    (f"Mconv2_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{i}_L1", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{i}_L1", [128, 38, 1, 1, 0]),
                ]
            )

            blocks[f"block{i}_2"] = OrderedDict(
                [
                    (f"Mconv1_stage{i}_L2", [185, 128, 7, 1, 3]),
                    (f"Mconv2_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{i}_L2", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{i}_L2", [128, 19, 1, 1, 0]),
                ]
            )

        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]

        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class handpose_model(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers have no relu layer
        no_relu_layers = [
            "conv6_2_CPM",
            "Mconv7_stage2",
            "Mconv7_stage3",
            "Mconv7_stage4",
            "Mconv7_stage5",
            "Mconv7_stage6",
        ]
        # stage 1
        block1_0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3", [512, 512, 3, 1, 1]),
                ("conv4_4", [512, 512, 3, 1, 1]),
                ("conv5_1", [512, 512, 3, 1, 1]),
                ("conv5_2", [512, 512, 3, 1, 1]),
                ("conv5_3_CPM", [512, 128, 3, 1, 1]),
            ]
        )

        block1_1 = OrderedDict(
            [("conv6_1_CPM", [128, 512, 1, 1, 0]), ("conv6_2_CPM", [512, 22, 1, 1, 0])]
        )

        blocks = {}
        blocks["block1_0"] = block1_0
        blocks["block1_1"] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks[f"block{i}"] = OrderedDict(
                [
                    (f"Mconv1_stage{i}", [150, 128, 7, 1, 3]),
                    (f"Mconv2_stage{i}", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{i}", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{i}", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{i}", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{i}", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{i}", [128, 22, 1, 1, 0]),
                ]
            )

        for k in blocks:
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks["block1_0"]
        self.model1_1 = blocks["block1_1"]
        self.model2 = blocks["block2"]
        self.model3 = blocks["block3"]
        self.model4 = blocks["block4"]
        self.model5 = blocks["block5"]
        self.model6 = blocks["block6"]

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


@lru_cache(maxsize=1)
def openpose_model():
    model = bodypose_model()
    weights_url = "https://huggingface.co/lllyasviel/ControlNet/resolve/38a62cbf79862c1bac73405ec8dc46133aee3e36/annotator/ckpts/body_pose_model.pth"
    model_path = get_cached_url_path(weights_url)
    model_dict = transfer(model, torch.load(model_path))
    model.load_state_dict(model_dict)
    model.eval()
    return model


def create_body_pose_img(original_img_t):
    candidate, subset = create_body_pose(original_img_t)
    canvas = np.zeros((original_img_t.shape[2], original_img_t.shape[3], 3))
    canvas = draw_bodypose(canvas, candidate, subset)
    canvas = torch.from_numpy(canvas).to(dtype=torch.float32)
    # canvas = canvas.unsqueeze(0)
    canvas = canvas.permute(2, 0, 1).unsqueeze(0)
    return canvas


def create_body_pose(original_img_t):
    original_img = torch_image_to_openvcv_img(original_img_t)

    model = openpose_model()
    # scale_search = [0.5, 1.0, 1.5, 2.0]
    scale_search = [0.5]
    boxsize = 368
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05
    multiplier = [x * boxsize / original_img.shape[0] for x in scale_search]
    heatmap_avg = np.zeros((original_img.shape[0], original_img.shape[1], 19))
    paf_avg = np.zeros((original_img.shape[0], original_img.shape[1], 38))

    for m, scale in enumerate(multiplier):
        imageToTest = cv2.resize(
            original_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        imageToTest_padded, pad = pad_right_down_corner(imageToTest, stride, padValue)
        im = (
            np.transpose(
                np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)
            )
            / 256
            - 0.5
        )
        im = np.ascontiguousarray(im)

        data = torch.from_numpy(im).float()
        data.to(get_device())

        # data = data.permute([2, 0, 1]).unsqueeze(0).float()
        with torch.no_grad():
            Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)
        Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

        # extract outputs, resize, and remove padding
        # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
        heatmap = np.transpose(
            np.squeeze(Mconv7_stage6_L2), (1, 2, 0)
        )  # output 1 is heatmaps
        heatmap = cv2.resize(
            heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
        )
        heatmap = heatmap[
            : imageToTest_padded.shape[0] - pad[2],
            : imageToTest_padded.shape[1] - pad[3],
            :,
        ]
        heatmap = cv2.resize(
            heatmap,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
        paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
        paf = cv2.resize(
            paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
        )
        paf = paf[
            : imageToTest_padded.shape[0] - pad[2],
            : imageToTest_padded.shape[1] - pad[3],
            :,
        ]
        paf = cv2.resize(
            paf,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap_avg += heatmap_avg + heatmap / len(multiplier)
        paf_avg += +paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (
                one_heatmap >= map_left,
                one_heatmap >= map_right,
                one_heatmap >= map_up,
                one_heatmap >= map_down,
                one_heatmap > thre1,
            )
        )
        peaks = list(
            zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
        )  # note reverse
        peaks_with_score = [(*x, map_ori[x[1], x[0]]) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [
            peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))
        ]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]
    # the middle joints heatmap correpondence
    mapIdx = [
        [31, 32],
        [39, 40],
        [33, 34],
        [35, 36],
        [41, 42],
        [43, 44],
        [19, 20],
        [21, 22],
        [23, 24],
        [25, 26],
        [27, 28],
        [29, 30],
        [47, 48],
        [49, 50],
        [53, 54],
        [51, 52],
        [55, 56],
        [37, 38],
        [45, 46],
    ]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)

                    startend = list(
                        zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num),
                        )
                    )

                    vec_x = np.array(
                        [
                            score_mid[
                                int(round(startend[I][1])),
                                int(round(startend[I][0])),
                                0,
                            ]
                            for I in range(len(startend))  # noqa
                        ]
                    )
                    vec_y = np.array(
                        [
                            score_mid[
                                int(round(startend[I][1])),
                                int(round(startend[I][0])),
                                1,
                            ]
                            for I in range(len(startend))  # noqa
                        ]
                    )

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(
                        vec_y, vec[1]
                    )
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * original_img.shape[0] / norm - 1, 0
                    )
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(
                        score_midpts
                    )
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [
                                i,
                                j,
                                score_with_dist_prior,
                                score_with_dist_prior + candA[i][2] + candB[j][2],
                            ]
                        )

            connection_candidate = sorted(
                connection_candidate, key=lambda x: x[2], reverse=True
            )
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack(
                        [connection, [candA[i][3], candB[j][3], s, i, j]]
                    )
                    if len(connection) >= min(nA, nB):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j, row in enumerate(subset):  # 1:size(subset,1):
                    if row[indexA] == partAs[i] or row[indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += (
                            candidate[partBs[i].astype(int), 2]
                            + connection_all[k][i][2]
                        )
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = (
                        (subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int)
                    )[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += (
                            candidate[partBs[i].astype(int), 2]
                            + connection_all[k][i][2]
                        )

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = (
                        sum(candidate[connection_all[k][i, :2].astype(int), 2])
                        + connection_all[k][i][2]
                    )
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = []

    for i, s in enumerate(subset):
        if s[-1] < 4 or s[-2] / s[-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
    # candidate: x, y, score, id
    return candidate, subset
