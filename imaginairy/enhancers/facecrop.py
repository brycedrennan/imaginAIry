import numpy as np

from imaginairy.enhancers.face_restoration_codeformer import face_restore_helper
from imaginairy.roi_utils import resize_roi_coordinates, square_roi_coordinate


def detect_faces(img):
    face_helper = face_restore_helper()
    face_helper.clean_all()

    image = img.convert("RGB")
    np_img = np.array(image, dtype=np.uint8)
    # rotate to BGR
    np_img = np_img[:, :, ::-1]

    face_helper.read_image(np_img)

    face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    face_helper.align_warp_face()
    faceboxes = []

    for x1, y1, x2, y2, scaling in face_helper.det_faces:
        # x1, y1, x2, y2 = x1 * scaling, y1 * scaling, x2 * scaling, y2 * scaling
        faceboxes.append((x1, y1, x2, y2))

    return faceboxes


def generate_face_crops(face_roi, max_width, max_height):
    """Returns bounding boxes at various zoom levels for faces in the image."""

    crops = []
    squared_roi = square_roi_coordinate(face_roi, max_width, max_height)

    crops.append(resize_roi_coordinates(squared_roi, 1.1, max_width, max_height))
    # 1.6 generally enough to capture entire face
    base_expanded_roi = resize_roi_coordinates(squared_roi, 1.6, max_width, max_height)

    crops.append(base_expanded_roi)
    current_width = base_expanded_roi[2] - base_expanded_roi[0]

    # some zoomed out variations
    for n in range(2):
        factor = 1.25 + 0.4 * n

        expanded_roi = resize_roi_coordinates(
            base_expanded_roi, factor, max_width, max_height, expand_up=False
        )
        new_width = expanded_roi[2] - expanded_roi[0]
        if new_width <= current_width * 1.1:
            # if the zoomed out size isn't suffienctly larger (because there is nowhere to zoom out to), stop
            break
        crops.append(expanded_roi)
        current_width = new_width
    return crops
