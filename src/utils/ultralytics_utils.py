import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate


def MaskImage(masked_image, bbox, mask_color: int | None = 0, edge_color: list[int] = [0, 0, 255]):
    """
    Mask the detected object with a black square and draw a red bounding box around the mask.

    Args:
        masked_image: (np.array) Image to be masked.
        bbox: (list) Bounding box coordinates [x1, y1, x2, y2].
        mask_color: (int) Color to mask the object with.
        edge_color: (list[int]) Color to draw the bounding box.

    Returns:
        (np.array) Image with the object masked and a bounding box around it.
    """
    masked_image = np.array(masked_image) if not isinstance(masked_image, np.ndarray) else masked_image
    # mask the detected object with a black square
    x1, y1, x2, y2 = map(int, bbox)
    if mask_color is not None:
        masked_image[y1:y2, x1:x2] = mask_color  # Mask the object with a black square

    # draw a red bounding box around the mask
    masked_image[y1:y2, x1 : x1 + 3] = edge_color  # Left vertical line
    masked_image[y1:y2, x2 : x2 + 3] = edge_color  # Right vertical line
    masked_image[y1 : y1 + 3, x1:x2] = edge_color  # Top horizontal line
    masked_image[y2 : y2 + 3, x1:x2] = edge_color  # Bottom horizontal line
    return masked_image


### Close set vocabs detection
def SAMYOLODetect(
    image_paths,  # (str): Path to a folder containing images to be annotated.
    model_path="/nas/pohan/models/",
    det_model="yolo11x.pt",
    sam_model="sam2.1_b.pt",
    device="1",
    conf=0.25,
    iou=0.45,
    output_dir="",
):
    # get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    os.chdir(model_path)
    # Ensure all image paths are absolute
    image_paths = [
        os.path.abspath(cwd + image_path) if not os.path.isabs(image_path) else image_path for image_path in image_paths
    ]

    # Perform auto annotation on the first image in the list
    # the result is saved to the output_dir with the same name as the image
    for image_path in image_paths:
        # Each line in the output text file represents a detected object with its class ID and segmentation points.
        # Class ID is from the COCO dataset with 80 classes, each with a unique ID ranging from 0 to 79
        # e.g., Class ID, point x (percentage), point y, point x, point y, ...
        auto_annotate(
            data=image_path,
            det_model=det_model,
            sam_model=sam_model,
            device=device,
            conf=conf,
            iou=iou,
            output_dir=output_dir,
        )


def OpenSetYOLODetect(
    image_paths,  # (list): List of image paths to be processed.
    class_names=["person", "cat", "truck", "car", "sedan"],  # (list): List of class names to be detected.
    model_path="/nas/pohan/models/",  # (str): Path to the directory containing the model.
    model_name="yolov8x-worldv2.pt",  # (str): Name of the pre-trained YOLO detection model: yolov8s/m/l/x-worldv2.pt
    save_results=True,  # (bool): Whether to save the prediction results.
    save_image_results=True,  # (bool): Whether to save the image results.
    device=1,  # (str): Specify the CUDA device, e.g., "cuda:0" or "cpu"
    output_dir=None,  # (str): Path to the directory to save the results.
):
    # get current working directory
    cwd = os.getcwd()

    os.chdir(model_path)
    # Initialize a YOLO-World model
    # Set the device to run the model
    if torch.cuda.is_available() and device != "cpu":
        device = f"cuda:{device}"
    else:
        device = "cpu"
    model = YOLO(model_name).to(device)

    # Define custom classes
    model.set_classes(class_names)
    # Save the model with the defined offline vocabulary
    model.save("custom_" + model_name)

    # Ensure all image paths are absolute
    image_paths = [
        os.path.abspath(cwd + image_path) if not os.path.isabs(image_path) else image_path for image_path in image_paths
    ]

    # if output_dir is not specified, save the results in the same directory as the input image
    if output_dir is None:
        output_dir = os.path.dirname(image_paths[0])

    # Execute prediction for specified categories on an image
    results = model.predict(image_paths)

    path2results = {}
    # Show results for each image
    for idx, result in enumerate(results):
        path2results[image_paths[idx]] = []

        # Access detection results
        detected_boxes = []
        masked_image = result.orig_img.copy()
        for box in result.boxes:
            detected_classes = result.names
            class_name = detected_classes[box.cls.item()]
            conf = box.conf.item()
            bbox = box.xyxy.detach().cpu().numpy().squeeze().tolist()
            detected_str = f"Class: {class_name}, Confidence: {conf}, BBox: {bbox}"
            detected_boxes.append(detected_str)

            if save_image_results:
                # Mask the detected object with a black square
                masked_image = MaskImage(masked_image, bbox)

        path2results[image_paths[idx]].append((class_name, conf, bbox))

        # Save results if needed
        if save_results:
            result_file_path = str(Path(output_dir)) + str(Path(image_paths[idx]).stem) + "_openset.txt"
            with open(result_file_path, "w") as f:
                for box_str in detected_boxes:
                    f.write(box_str + "\n")
            print("Results saved successfully.")

        if save_image_results:
            # Save the visualization of the results to a jpg file
            result_image_path = str(Path(output_dir)) + str(Path(image_paths[idx]).stem) + "_od.jpg"
            result.save(result_image_path)

            # Save the masked image
            masked_image_path = str(Path(output_dir)) + str(Path(image_paths[idx]).stem) + "_masked.jpg"
            masked_image = masked_image[:, :, ::-1]  # Convert RGB to BGR, only for Yolo
            masked_image_pil = Image.fromarray(masked_image)
            masked_image_pil.save(masked_image_path)

    return


if __name__ == "__main__":
    # Example usage
    image_paths = [
        "/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/keyframe_005.jpg",
        "/home/ta-tldr/Project/tldr/keyframes/3_scene_detect/3_keyframe_1.jpg",
    ]
    # SAMYOLODetect(image_paths, output_dir="/home/ta-tldr/Project/tldr/")
    OpenSetYOLODetect(image_paths, output_dir="/home/ta-tldr/Project/tldr/")
