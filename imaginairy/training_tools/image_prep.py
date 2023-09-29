import logging
import os
import os.path
import re

from PIL import Image
from tqdm import tqdm

from imaginairy import ImaginePrompt, LazyLoadingImage, imagine
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.enhancers.facecrop import detect_faces, generate_face_crops
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from imaginairy.vendored.smart_crop import SmartCrop

logger = logging.getLogger(__name__)


def get_image_filenames(folder):
    filenames = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if filename.startswith("."):
            continue
        filenames.append(filename)
    return filenames


def prep_images(
    images_dir, is_person=False, output_folder_name="prepped-images", target_size=512
):
    """
    Crops and resizes a directory of images in preparation for training.

    If is_person=True, it will detect the face and produces several crops at different zoom levels. For crops that
    are too small, it will use the face restoration model to enhance the faces.

    For non-person images, it will use the smartcrop algorithm to crop the image to the most interesting part. If the
    input image is too small it will be upscaled.

    Prep will go a lot faster if all the images are big enough to not require upscaling.

    """
    output_folder = os.path.join(images_dir, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Prepping images in {images_dir} to {output_folder}")
    image_filenames = get_image_filenames(images_dir)
    pbar = tqdm(image_filenames)
    for filename in pbar:
        pbar.set_description(filename)

        input_path = os.path.join(images_dir, filename)
        img = LazyLoadingImage(filepath=input_path).convert("RGB")
        if is_person:
            face_rois = detect_faces(img)
            if len(face_rois) == 0:
                logger.info(f"No faces detected in image {filename}, skipping")
                continue
            if len(face_rois) > 1:
                logger.info(f"Multiple faces detected in image {filename}, skipping")
                continue
            face_roi = face_rois[0]
            face_roi_crops = generate_face_crops(
                face_roi, max_width=img.width, max_height=img.height
            )
            for n, face_roi_crop in enumerate(face_roi_crops):
                cropped_output_path = os.path.join(
                    output_folder, f"{filename}_[alt-{n:02d}].jpg"
                )
                if os.path.exists(cropped_output_path):
                    logger.debug(
                        f"Skipping {cropped_output_path} because it already exists"
                    )
                    continue
                x1, y1, x2, y2 = face_roi_crop
                crop_width = x2 - x1
                crop_height = y2 - y1
                if crop_width != crop_height:
                    logger.info(
                        f"Face ROI crop for {filename} {crop_width}x{crop_height} is not square, skipping"
                    )
                    continue
                cropped_img = img.crop(face_roi_crop)

                if crop_width < target_size:
                    logger.info(f"Upscaling {filename} {face_roi_crop}")
                    cropped_img = cropped_img.resize(
                        (target_size, target_size), resample=Image.Resampling.LANCZOS
                    )
                    cropped_img = enhance_faces(cropped_img, fidelity=1)
                else:
                    cropped_img = cropped_img.resize(
                        (target_size, target_size), resample=Image.Resampling.LANCZOS
                    )
                cropped_img.save(cropped_output_path, quality=95)
        else:
            # scale image so that largest dimension is target_size
            n = 0
            cropped_output_path = os.path.join(output_folder, f"{filename}_{n}.jpg")
            if os.path.exists(cropped_output_path):
                logger.debug(
                    f"Skipping {cropped_output_path} because it already exists"
                )
                continue
            if img.width < target_size or img.height < target_size:
                # upscale the image if it's too small
                logger.info(f"Upscaling {filename}")
                img = upscale_image(img)

            if img.width > img.height:
                scale_factor = target_size / img.height
            else:
                scale_factor = target_size / img.width

            # downscale so shortest side is target_size
            new_width = int(round(img.width * scale_factor))
            new_height = int(round(img.height * scale_factor))
            img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

            result = SmartCrop().crop(img, width=target_size, height=target_size)

            box = (
                result["top_crop"]["x"],
                result["top_crop"]["y"],
                result["top_crop"]["width"] + result["top_crop"]["x"],
                result["top_crop"]["height"] + result["top_crop"]["y"],
            )

            cropped_image = img.crop(box)
            cropped_image.save(cropped_output_path, quality=95)
    logger.info(f"Image Prep complete. Review output at {output_folder}")


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,\[\]-]+", "-", prompt)[:130]


def create_class_images(class_description, output_folder, num_images=200):
    """
    Generate images of class_description.
    """
    existing_images = get_image_filenames(output_folder)
    existing_image_count = len(existing_images)
    class_slug = prompt_normalized(class_description)

    while existing_image_count < num_images:
        prompt = ImaginePrompt(class_description, steps=20)
        result = next(iter(imagine([prompt])))
        if result.is_nsfw:
            continue
        dest = os.path.join(
            output_folder, f"{existing_image_count:03d}_{class_slug}.jpg"
        )
        result.save(dest)
        existing_image_count += 1
