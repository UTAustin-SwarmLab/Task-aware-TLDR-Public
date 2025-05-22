import os
import re
import sys
from pathlib import Path

import easyocr
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.ultralytics_utils import MaskImage

# Initialize lemmatizer for combining singular/plural forms
lemmatizer = WordNetLemmatizer()

# Download required NLTK data if not already available
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")


def EasyOCR(
    image_paths: list[str] | list[Path],
    postfix: str,
    keywords: None | list[str] = None,
    save_results=False,
    save_image_results=True,
    languages=["en"],
    conf_threshold: float = 0.4,
    output_dir: None | str | Path = None,
    reader: None | easyocr.Reader = None,
):
    """
    Perform Optical Character Recognition (OCR) on images using EasyOCR and masked the bbox with keywords.

    Args:
        image_paths: (list) List of image paths to perform OCR on.
        postfix: (str) Postfix to append to the output files.
        keywords: (list) List of keywords to mask the detected text with. If None, all text is masked.
        save_results: (bool) Whether to save the OCR results to a csv file.
        save_image_results: (bool) Whether to save the masked image to a jpg file.
        languages: (list) List of languages to use for OCR.
        conf_threshold: (float) Minimum confidence level to consider a detection.
        output_dir: (str) Directory to save the results in.
    """
    if reader is None:
        reader = easyocr.Reader(languages)  # this needs to run only once to load the model into memory
    if not isinstance(output_dir, str):
        output_dir = str(output_dir)

    # get current working directory
    cwd = os.getcwd()
    # if output_dir is not specified, save the results in the same directory as the input image
    if output_dir is None:
        output_dir = os.path.dirname(image_paths[0])
    else:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir + f"/textod{postfix}").mkdir(parents=True, exist_ok=True)
        Path(output_dir + f"/textmasked{postfix}").mkdir(parents=True, exist_ok=True)
        Path(output_dir + f"/slidetext{postfix}").mkdir(parents=True, exist_ok=True)

    # Ensure all image paths are absolute
    image_paths = [
        os.path.abspath(cwd + str(image_path)) if not os.path.isabs(str(image_path)) else str(image_path)
        for image_path in image_paths
    ]

    # Preprocess keywords if provided to include both singular and plural forms
    lemmatized_keywords = None
    if keywords is not None:
        # Create a set of all lemmatized keywords
        lemmatized_keywords = set()
        for keyword in keywords:
            # Add original keyword
            lemmatized_keywords.add(keyword.lower())
            # Add lemmatized form (singular)
            lemma = lemmatizer.lemmatize(keyword.lower(), pos="n")
            lemmatized_keywords.add(lemma)
            # Try to create plural form by adding 's' and add it too
            # This is a simple heuristic and won't work for all words
            if keyword.lower() == lemma:  # If word is already in singular form
                lemmatized_keywords.add(f"{lemma}s")
        # Convert back to list
        lemmatized_keywords = list(lemmatized_keywords)

    for image_path in image_paths:
        result = reader.readtext(image_path)
        result_image = Image.open(image_path)
        masked_image = result_image.copy()
        # bbox_text_conf = []

        result_txt_path = output_dir + f"/slidetext{postfix}/" + str(Path(image_path).stem) + "_text.csv"
        # Write header only when creating the file
        with open(result_txt_path, "w") as f:
            f.write("bbox,text,conf\n")

        for bbox, text, conf in result:
            if conf >= conf_threshold:
                # bbox_text_conf.append((bbox, text, conf))
                bbox = np.array(bbox, dtype=int)
                x1, x2 = bbox[0][0], bbox[2][0]
                y1, y2 = bbox[0][1], bbox[2][1]
                bbox = [x1, y1, x2, y2]
                # print(f"Detected text: {text}, with probability: {conf}, at bbox: {bbox}")
                # Save results if needed
                if save_results:
                    with open(result_txt_path, "a") as f:
                        # Append the current detection
                        f.write(f"{bbox},{text},{conf}\n")

                if save_image_results:
                    mask_TF = lemmatized_keywords is None
                    if not mask_TF:
                        # Lemmatize words in the detected text
                        text_words = re.findall(r"\b\w+\b", text.lower())
                        lemmatized_text_words = [lemmatizer.lemmatize(word, pos="n") for word in text_words]

                        for keyword in lemmatized_keywords:
                            # Check if any lemmatized word in text matches the keyword
                            if any(
                                re.search(rf"\b{re.escape(keyword)}\b", word, re.IGNORECASE)
                                for word in text_words + lemmatized_text_words
                            ):
                                mask_TF = True
                                break
                    if mask_TF:
                        # Save the visualization of the results to a jpg file
                        result_image_path = (
                            output_dir + f"/textod{postfix}/" + str(Path(image_path).stem) + "_textod.jpg"
                        )
                        result_image = MaskImage(result_image, bbox, mask_color=None, edge_color=[255, 0, 0])
                        result_image_pil = Image.fromarray(result_image)
                        result_image_pil.save(result_image_path)

                        # Save the masked image
                        masked_image_path = (
                            output_dir + f"/textmasked{postfix}/" + str(Path(image_path).stem) + "_textmasked.jpg"
                        )
                        masked_image = MaskImage(masked_image, bbox, edge_color=[255, 0, 0])
                        masked_image_pil = Image.fromarray(masked_image)
                        masked_image_pil.save(masked_image_path)
    return


if __name__ == "__main__":
    video_root = "/nas/pohan/datasets/AIConfVideo/video/39024853/"
    # "/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/keyframe_005.jpg",
    # Example usage
    image_paths = [video_root + f"/slide_{i}.png" for i in range(5, 10)]
    EasyOCR(
        image_paths,
        keywords=["unitary", "damped", "node"],
        save_results=True,
        save_image_results=True,
        output_dir="/home/ta-tldr/Project/tldr/",
    )
