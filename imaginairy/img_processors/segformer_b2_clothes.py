from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    from torch import Tensor  # noqa

ade20_palette = [
    [255, 0, 0],  # red
    [255, 255, 0],  # yellow
    [0, 255, 0],  # green
    [0, 0, 255],  # blue
    [0, 128, 0],  # dark green
    [128, 0, 128],  # purple
    [255, 165, 0],  # orange
    [139, 69, 19],  # brown
    [30, 144, 255],  # dodger blue
    [123, 104, 238],  # medium slate blue
    [255, 0, 255],  # magenta
    [255, 192, 203],  # pink
    [112, 128, 144],  # slate gray
    [218, 165, 32],  # goldenrod
    [128, 128, 0],  # olive
    [0, 128, 128],  # teal
    [148, 0, 211],  # dark violet
    [205, 92, 92],  # indian red
]

simple_palette = [
    [200, 200, 200],
    [255, 192, 203],
    [0, 128, 0],
]


@dataclass
class B2Segment:
    id: int
    name: str
    simple_name: Literal["background", "body", "cloth"]


@dataclass
class B2SegmentMap:
    logits: "Tensor"
    img: "Image.Image"

    @cached_property
    def segmap(self) -> "Tensor":
        import torch

        return torch.argmax(self.logits, dim=-1)[0]

    @staticmethod
    def segmap_to_color(segmap: "Tensor", palette=ade20_palette) -> "np.ndarray":
        import numpy as np

        color_seg = np.zeros(
            (segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8
        )  # height, width, 3
        for segment_id, color in enumerate(palette):
            color_seg[segmap.numpy() == segment_id, :] = color
        return color_seg

    @cached_property
    def segmap_color(self) -> "np.ndarray":
        return self.segmap_to_color(self.segmap)

    @cached_property
    def simple_segmap(self) -> "Tensor":
        """Return a simplified segmentation map with only 3 classes: background, cloth, body."""
        segmap = self.segmap
        simple_segmap = segmap.clone()
        for segment in segments:
            if segment.simple_name == "background":
                simple_segmap[segmap == segment.id] = 0
            elif segment.simple_name == "body":
                simple_segmap[segmap == segment.id] = 1
            elif segment.simple_name == "cloth":
                simple_segmap[segmap == segment.id] = 2
        return simple_segmap

    @cached_property
    def simple_segmap_color(self) -> "np.ndarray":
        return self.segmap_to_color(self.simple_segmap, palette=simple_palette)

    def make_clothing_segmentation_diagram(self) -> "Image.Image":
        segment_names = [s.name for s in segments]
        return make_clothing_segmentation_diagram(
            self.img,
            self.segmap,
            color_seg=self.segmap_color,
            segment_names=segment_names,
            palette=ade20_palette,
        )

    def make_simple_clothing_segmentation_diagram(self) -> "Image.Image":
        segment_names = ["background", "body", "cloth"]
        return make_clothing_segmentation_diagram(
            self.img,
            self.simple_segmap,
            color_seg=self.simple_segmap_color,
            segment_names=segment_names,
            palette=simple_palette,
        )


segments = [
    B2Segment(0, "Background", "background"),
    B2Segment(1, "Hat", "cloth"),
    B2Segment(2, "Hair", "body"),
    B2Segment(3, "Sunglasses", "cloth"),
    B2Segment(4, "Upper-clothes", "cloth"),
    B2Segment(5, "Skirt", "cloth"),
    B2Segment(6, "Pants", "cloth"),
    B2Segment(7, "Dress", "cloth"),
    B2Segment(8, "Belt", "cloth"),
    B2Segment(9, "Left-shoe", "cloth"),
    B2Segment(10, "Right-shoe", "cloth"),
    B2Segment(11, "Face", "body"),
    B2Segment(12, "Left-leg", "body"),
    B2Segment(13, "Right-leg", "body"),
    B2Segment(14, "Left-arm", "body"),
    B2Segment(15, "Right-arm", "body"),
    B2Segment(16, "Bag", "cloth"),
    B2Segment(17, "Scarf", "cloth"),
]


@lru_cache(maxsize=1)
def segmodel():
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    repo_id = "mattmdjaga/segformer_b2_clothes"
    processor = SegformerImageProcessor.from_pretrained(repo_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(repo_id)
    return processor, model


def make_clothing_segmentation_map(image_pil) -> B2SegmentMap:
    import pillow_avif  # noqa
    import torch.nn as nn

    processor, model = segmodel()

    inputs = processor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    logits = nn.functional.interpolate(
        logits,
        size=image_pil.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    logits = logits.permute(0, 2, 3, 1)
    return B2SegmentMap(logits, image_pil)


def label_to_color_image(label, palette):
    import numpy as np

    colormap = np.asarray(palette)
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")
    return colormap[label]


def make_clothing_segmentation_diagram(
    image_pil, seg, color_seg, segment_names, palette: list
) -> "Image.Image":
    import io

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec
    from PIL import Image

    pred_img = np.array(image_pil) * 0.4 + color_seg * 0.6
    pred_img = pred_img.astype(np.uint8)

    fig = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0, top=1, bottom=0, wspace=0.06, hspace=0)

    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(pred_img)
    plt.axis("off")
    segment_names = np.asarray(segment_names)
    FULL_LABEL_MAP = np.arange(len(segment_names)).reshape(len(segment_names), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, palette=palette)

    unique_labels = np.unique(seg.numpy().astype("uint8"))
    ax = plt.subplot(grid_spec[1])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation="nearest")
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), segment_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0, labelsize=25)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    chart_img = Image.open(buf)

    return chart_img


def demo(img_path):
    import pillow_avif  # noqa
    from PIL import Image

    image = Image.open(img_path).convert("RGB")

    seg = make_clothing_segmentation_map(image)
    Image.fromarray(seg.segmap_color).show()
    seg.make_clothing_segmentation_diagram().show()

    seg.make_simple_clothing_segmentation_diagram().show()
