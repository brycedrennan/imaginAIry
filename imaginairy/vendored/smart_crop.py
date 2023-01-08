"""
Crops to the most interesting part of the image.

MIT License from https://github.com/smartcrop/smartcrop.py/commit/f5377045035abc7ae79d8d9ad40bbc7fce0f6ad7
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageFilter import Kernel


def saturation(image):
    r, g, b = image.split()
    r, g, b = np.array(r), np.array(g), np.array(b)
    r, g, b = r.astype(float), g.astype(float), b.astype(float)
    maximum = np.maximum(np.maximum(r, g), b)  # [0; 255]
    minimum = np.minimum(np.minimum(r, g), b)  # [0; 255]
    s = (maximum + minimum) / 255  # [0.0; 1.0]
    d = (maximum - minimum) / 255  # [0.0; 1.0]
    d[maximum == minimum] = 0  # if maximum == minimum:
    s[maximum == minimum] = 1  # -> saturation = 0 / 1 = 0
    mask = s > 1
    s[mask] = 2 - d[mask]
    return d / s  # [0.0; 1.0]


def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]."""
    x = ((x + 2 / 3) % 2 * 0.5 - 0.5) * 16
    return max(1 - x * x, 0)


class SmartCrop:
    DEFAULT_SKIN_COLOR = [0.78, 0.57, 0.44]

    def __init__(
        self,
        detail_weight=0.2,
        edge_radius=0.4,
        edge_weight=-20,
        outside_importance=-0.5,
        rule_of_thirds=True,
        saturation_bias=0.2,
        saturation_brightness_max=0.9,
        saturation_brightness_min=0.05,
        saturation_threshold=0.4,
        saturation_weight=0.3,
        score_down_sample=8,
        skin_bias=0.01,
        skin_brightness_max=1,
        skin_brightness_min=0.2,
        skin_color=None,
        skin_threshold=0.8,
        skin_weight=1.8,
    ):
        self.detail_weight = detail_weight
        self.edge_radius = edge_radius
        self.edge_weight = edge_weight
        self.outside_importance = outside_importance
        self.rule_of_thirds = rule_of_thirds
        self.saturation_bias = saturation_bias
        self.saturation_brightness_max = saturation_brightness_max
        self.saturation_brightness_min = saturation_brightness_min
        self.saturation_threshold = saturation_threshold
        self.saturation_weight = saturation_weight
        self.score_down_sample = score_down_sample
        self.skin_bias = skin_bias
        self.skin_brightness_max = skin_brightness_max
        self.skin_brightness_min = skin_brightness_min
        self.skin_color = skin_color or self.DEFAULT_SKIN_COLOR
        self.skin_threshold = skin_threshold
        self.skin_weight = skin_weight

    def analyse(
        self,
        image,
        crop_width,
        crop_height,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        """
        Analyze image and return some suggestions of crops (coordinates).
        This implementation / algorithm is really slow for large images.
        Use `crop()` which is pre-scaling the image before analyzing it.
        """
        cie_image = image.convert("L", (0.2126, 0.7152, 0.0722, 0))
        cie_array = np.array(cie_image)  # [0; 255]

        # R=skin G=edge B=saturation
        edge_image = self.detect_edge(cie_image)
        skin_image = self.detect_skin(cie_array, image)
        saturation_image = self.detect_saturation(cie_array, image)
        analyse_image = Image.merge("RGB", [skin_image, edge_image, saturation_image])

        del edge_image
        del skin_image
        del saturation_image

        score_image = analyse_image.copy()
        score_image.thumbnail(
            (
                int(math.ceil(image.size[0] / self.score_down_sample)),
                int(math.ceil(image.size[1] / self.score_down_sample)),
            ),
            Image.ANTIALIAS,
        )

        top_crop = None
        top_score = -sys.maxsize

        crops = self.crops(
            image,
            crop_width,
            crop_height,
            max_scale=max_scale,
            min_scale=min_scale,
            scale_step=scale_step,
            step=step,
        )

        for crop in crops:
            crop["score"] = self.score(score_image, crop)
            if crop["score"]["total"] > top_score:
                top_crop = crop
                top_score = crop["score"]["total"]

        return {"analyse_image": analyse_image, "crops": crops, "top_crop": top_crop}

    def crop(
        self,
        image,
        width,
        height,
        prescale=True,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        """Not yet fully cleaned from https://github.com/hhatto/smartcrop.py."""
        scale = min(image.size[0] / width, image.size[1] / height)
        crop_width = int(math.floor(width * scale))
        crop_height = int(math.floor(height * scale))
        # img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
        # don't set minscale smaller than 1/scale
        # -> don't pick crops that need upscaling
        min_scale = min(max_scale, max(1 / scale, min_scale))

        prescale_size = 1
        if prescale:
            prescale_size = 1 / scale / min_scale
            if prescale_size < 1:
                image = image.copy()
                image.thumbnail(
                    (
                        int(image.size[0] * prescale_size),
                        int(image.size[1] * prescale_size),
                    ),
                    Image.ANTIALIAS,
                )
                crop_width = int(math.floor(crop_width * prescale_size))
                crop_height = int(math.floor(crop_height * prescale_size))
            else:
                prescale_size = 1

        result = self.analyse(
            image,
            crop_width=crop_width,
            crop_height=crop_height,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
            step=step,
        )

        for i in range(len(result["crops"])):
            crop = result["crops"][i]
            crop["x"] = int(math.floor(crop["x"] / prescale_size))
            crop["y"] = int(math.floor(crop["y"] / prescale_size))
            crop["width"] = int(math.floor(crop["width"] / prescale_size))
            crop["height"] = int(math.floor(crop["height"] / prescale_size))
            result["crops"][i] = crop
        return result

    def crops(
        self,
        image,
        crop_width,
        crop_height,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        image_width, image_height = image.size
        crops = []
        for scale in (
            i / 100
            for i in range(
                int(max_scale * 100),
                int((min_scale - scale_step) * 100),
                -int(scale_step * 100),
            )
        ):
            for y in range(0, image_height, step):
                if not (y + crop_height * scale <= image_height):
                    break
                for x in range(0, image_width, step):
                    if not (x + crop_width * scale <= image_width):
                        break
                    crops.append(
                        {
                            "x": x,
                            "y": y,
                            "width": crop_width * scale,
                            "height": crop_height * scale,
                        }
                    )
        if not crops:
            raise ValueError(locals())
        return crops

    def debug_crop(self, analyse_image, crop):
        debug_image = analyse_image.copy()
        debug_pixels = debug_image.getdata()
        debug_crop_image = Image.new(
            "RGBA",
            (int(math.floor(crop["width"])), int(math.floor(crop["height"]))),
            (255, 0, 0, 25),
        )
        ImageDraw.Draw(debug_crop_image).rectangle(
            ((0, 0), (crop["width"], crop["height"])), outline=(255, 0, 0)
        )

        for y in range(analyse_image.size[1]):  # height
            for x in range(analyse_image.size[0]):  # width
                p = y * analyse_image.size[0] + x
                importance = self.importance(crop, x, y)
                if importance > 0:
                    debug_pixels.putpixel(
                        (x, y),
                        (
                            debug_pixels[p][0],
                            int(debug_pixels[p][1] + importance * 32),
                            debug_pixels[p][2],
                        ),
                    )
                elif importance < 0:
                    debug_pixels.putpixel(
                        (x, y),
                        (
                            int(debug_pixels[p][0] + importance * -64),
                            debug_pixels[p][1],
                            debug_pixels[p][2],
                        ),
                    )
        debug_image.paste(
            debug_crop_image, (crop["x"], crop["y"]), debug_crop_image.split()[3]
        )
        return debug_image

    def detect_edge(self, cie_image):
        return cie_image.filter(Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 1))

    def detect_saturation(self, cie_array, source_image):
        threshold = self.saturation_threshold
        saturation_data = saturation(source_image)
        mask = (
            (saturation_data > threshold)
            & (cie_array >= self.saturation_brightness_min * 255)
            & (cie_array <= self.saturation_brightness_max * 255)
        )

        saturation_data[~mask] = 0
        saturation_data[mask] = (saturation_data[mask] - threshold) * (
            255 / (1 - threshold)
        )

        return Image.fromarray(saturation_data.astype("uint8"))

    def detect_skin(self, cie_array, source_image):
        r, g, b = source_image.split()
        r, g, b = np.array(r), np.array(g), np.array(b)
        r, g, b = r.astype(float), g.astype(float), b.astype(float)
        rd = np.ones_like(r) * -self.skin_color[0]
        gd = np.ones_like(g) * -self.skin_color[1]
        bd = np.ones_like(b) * -self.skin_color[2]

        mag = np.sqrt(r * r + g * g + b * b)
        mask = ~(abs(mag) < 1e-6)
        rd[mask] = r[mask] / mag[mask] - self.skin_color[0]
        gd[mask] = g[mask] / mag[mask] - self.skin_color[1]
        bd[mask] = b[mask] / mag[mask] - self.skin_color[2]

        skin = 1 - np.sqrt(rd * rd + gd * gd + bd * bd)
        mask = (
            (skin > self.skin_threshold)
            & (cie_array >= self.skin_brightness_min * 255)
            & (cie_array <= self.skin_brightness_max * 255)
        )

        skin_data = (skin - self.skin_threshold) * (255 / (1 - self.skin_threshold))
        skin_data[~mask] = 0

        return Image.fromarray(skin_data.astype("uint8"))

    def importance(self, crop, x, y):
        if (
            crop["x"] > x
            or x >= crop["x"] + crop["width"]
            or crop["y"] > y
            or y >= crop["y"] + crop["height"]
        ):
            return self.outside_importance

        x = (x - crop["x"]) / crop["width"]
        y = (y - crop["y"]) / crop["height"]
        px, py = abs(0.5 - x) * 2, abs(0.5 - y) * 2

        # distance from edge
        dx = max(px - 1 + self.edge_radius, 0)
        dy = max(py - 1 + self.edge_radius, 0)
        d = (dx * dx + dy * dy) * self.edge_weight
        s = 1.41 - math.sqrt(px * px + py * py)

        if self.rule_of_thirds:
            s += (max(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py))

        return s + d

    def score(self, target_image, crop):
        score = {
            "detail": 0,
            "saturation": 0,
            "skin": 0,
            "total": 0,
        }
        target_data = target_image.getdata()
        target_width, target_height = target_image.size

        down_sample = self.score_down_sample
        inv_down_sample = 1 / down_sample
        target_width_down_sample = target_width * down_sample
        target_height_down_sample = target_height * down_sample

        for y in range(0, target_height_down_sample, down_sample):
            for x in range(0, target_width_down_sample, down_sample):
                p = int(
                    math.floor(y * inv_down_sample) * target_width
                    + math.floor(x * inv_down_sample)
                )
                importance = self.importance(crop, x, y)
                detail = target_data[p][1] / 255
                score["skin"] += (
                    target_data[p][0] / 255 * (detail + self.skin_bias) * importance
                )
                score["detail"] += detail * importance
                score["saturation"] += (
                    target_data[p][2]
                    / 255
                    * (detail + self.saturation_bias)
                    * importance
                )
        score["total"] = (
            score["detail"] * self.detail_weight
            + score["skin"] * self.skin_weight
            + score["saturation"] * self.saturation_weight
        ) / (crop["width"] * crop["height"])
        return score
