import json
import os
import shutil

import cv2
import numpy as np
import openai
from skimage import metrics

from imaginairy import LazyLoadingImage
from imaginairy.enhancers.describe_image_blip import generate_caption


def describe_video(video_path, delete_frames=True, frames_directory="key_frames"):
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    if not openai.api_key:
        raise KeyError("OPENAI_API_KEY environment variable not set")

    # Extract key frames from the video
    key_frames = extract_key_frames(
        video_path, threshold=0.01, key_frames_dir=frames_directory
    )
    # Generate descriptions for each key frame
    frame_descriptions = []
    for frame_idx, timestamp, frame_path in key_frames:
        description = describe_frame(frame_path)
        frame_descriptions.append(description)

    # Submit descriptions to OpenAI API

    setting_description = ""
    video_summary = ""
    # frame_descriptions = ""

    #

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    for chunk in chunker(frame_descriptions, 50):
        descriptions_chunk = chunk

        prompt = f"""
        I want you to summarize what's happened in a video, based of the description of key frames given to you in chronological order, in batches.
        The key frames are frames where something changed in the video. Please tell me the description of what the camera sees, along with a summary of events or changes.
        The response should be in the form of json, with keys for the setting_description and video_summary. You may recieve this information in batches, so add on to what will
        be given to you, which will the answers you gave me in previous queries. environment_description is simply a description of what the setting of the video. video_summary is a summary of
        what changes have happened in this setting. setting_description is a description of the setting, the non changing aspects of what the camera is seeing. frame_descriptions here are more
        key frame descriptions that you can use to add descriptions to summary and setting_description.

        setting_description: {setting_description}

        summary: {video_summary}

        frame_descriptions: {descriptions_chunk}
    """

        completion = openai.ChatCompletion.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )

        response = json.loads(completion["choices"][0]["message"]["content"])

        video_summary = response["video_summary"]
        setting_description = response["setting_description"]

    summary = completion["choices"][0]["message"]["content"]

    if delete_frames:
        shutil.rmtree(frames_directory)

    return summary


def describe_frame(frame):
    img = LazyLoadingImage(filepath=frame)
    caption = generate_caption(img.copy())
    return caption


def extract_key_frames(
    video_path, threshold=0.01, prune_frames=True, key_frames_dir="key_frames"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file.")

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise Exception("Error reading video file.")

    if not os.path.exists(key_frames_dir):
        os.makedirs(key_frames_dir)

    key_frames = []  # Add the first frame to the key frames list
    frame_idx = 1
    # root mean square

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # Calculate frame difference
        rmse = difference_between_images(prev_frame, current_frame)

        # Compare with threshold
        if rmse > threshold:
            # Calculate the timestamp for the key frame
            timestamp = frame_idx / fps

            # Write the key frame to a file
            key_frame_path = os.path.join(key_frames_dir, f"key_frame_{frame_idx}.jpg")
            cv2.imwrite(key_frame_path, current_frame)

            # Add the key frame to the list
            key_frames.append((frame_idx, timestamp, key_frame_path))

        prev_frame = current_frame
        frame_idx += 1

    cap.release()
    # reduces the number of frames by removing the number of highly similiar frames.
    if prune_frames:
        key_frames = find_sequences(key_frames)

    # if delete_files:
    #     shutil.rmtree(key_frames_dir)

    return key_frames


def test_extract_key_frames():
    assert len(extract_key_frames(video_path="test_security_feed.mp4")) == 83
    assert (
        len(extract_key_frames(video_path="test_security_feed.mp4", prune_frames=False))
        == 197
    )


def find_sequences(lst):
    if not lst:
        return

    sequences = [[lst[0]]]

    for i in range(1, len(lst)):
        if lst[i][0] - lst[i - 1][0] == 1:
            sequences[-1].append(lst[i])
        else:
            sequences.append([lst[i]])

    # process sequences to keep 1 out of every 4 elements
    output = []
    for seq in sequences:
        if len(seq) >= 4:
            subset = [seq[j] for j in range(0, len(seq), 4)]
            output.extend(subset)
        else:
            output.extend(seq)

    # modify the original list with the output
    lst.clear()
    lst.extend(output)
    return lst


def test_find_sequences():
    # Test case 1: empty list
    assert find_sequences([]) == []

    # Test case 2: list with one element
    assert find_sequences([(0, "a")]) == [(0, "a")]

    # Test case 3: list with no consecutive elements
    assert find_sequences([(0, "a"), (2, "b"), (4, "c")]) == [
        (0, "a"),
        (2, "b"),
        (4, "c"),
    ]

    # Test case 4: list with consecutive elements
    assert find_sequences([(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]) == [
        (0, "a"),
        (4, "e"),
    ]

    # Test case 5: list with consecutive elements and remainder
    assert find_sequences(
        [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e"), (5, "f")]
    ) == [(0, "a"), (4, "e")]

    # Test case 6: list with consecutive elements and two remainder
    assert find_sequences(
        [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e"), (5, "f")]
    ) == [(0, "a"), (4, "e")]

    # Test case 7: list with multiple short sequences
    assert find_sequences(
        [(0, "a"), (1, "b"), (2, "c"), (4, "d"), (5, "e"), (6, "f"), (8, "g"), (9, "h")]
    ) == (
        [(0, "a"), (1, "b"), (2, "c"), (4, "d"), (5, "e"), (6, "f"), (8, "g"), (9, "h")]
    )

    # Test case 8: list with multiple short and long sequences
    assert find_sequences(
        [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (5, "e"),
            (6, "f"),
            (8, "g"),
            (9, "h"),
            (10, "i"),
            (11, "j"),
            (12, "k"),
            (13, "l"),
            (14, "m"),
        ]
    ) == [(0, "a"), (5, "e"), (6, "f"), (8, "g"), (12, "k")]


def difference_between_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate mean squared error
    mse = metrics.mean_squared_error(gray1, gray2)

    # Calculate the maximum possible MSE value
    max_mse = np.max(gray1) ** 2

    # Calculate the minimum possible MSE value
    min_mse = 0

    # Normalize the MSE value to a range of 0 to 100
    if max_mse != min_mse:
        normalized_mse = 100 * (mse - min_mse) / (max_mse - min_mse)
    else:
        normalized_mse = 0
    return normalized_mse


def test_difference_between_images_identical_images():
    # Load two identical test images
    image1 = cv2.imread("assets/pearl_depth_2.jpg")
    image2 = cv2.imread("assets/pearl_depth_2.jpg")

    # Calculate the difference between the images
    mse_actual = difference_between_images(image1, image2)

    # Check that the calculated MSE is 0
    assert mse_actual == 0


def test_difference_between_images_different_images():
    # Load two different test images
    image1 = cv2.imread("assets/pearl_depth_2.jpg")
    image2 = cv2.imread("assets/pearl_depth_1.jpg")

    # Calculate the difference between the images
    mse_actual = difference_between_images(image1, image2)

    # Check that the calculated MSE is greater than 0
    assert mse_actual > 0


# describe_video(video_path="test_security_feed.mp4")
