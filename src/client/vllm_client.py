# CUDA_VISIBLE_DEVICES=1 python src/client/vllm_client.py
import math
import os
import random
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from vllm import LLM, SamplingParams

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from src.client.openai_API_client import OpenAIVllmClient


def crop_image(image: Path | str | Image.Image, grids: int = 5):
    """
    Crop the image into grids x grids pieces. and return a list of cropped images.
    """
    # convert to numpy array if image is PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    # convert to numpy array if image is a path
    if isinstance(image, str) or isinstance(image, Path):
        image = cv2.imread(image)
    height, width = image.shape[:2]
    grid_height = height // grids
    grid_width = width // grids

    cropped_images = []
    for i in range(grids):
        for j in range(grids):
            x_start = i * grid_height
            y_start = j * grid_width
            x_end = (i + 1) * grid_height
            y_end = (j + 1) * grid_width
            cropped_images.append(image[x_start:x_end, y_start:y_end])
    return cropped_images


def parse_selection_token_prob(response_choice):
    parenthesis_flag = False
    for logprob_entry in response_choice.logprobs:
        # tokenID = list(logprob_entry.keys())[0]
        logprobItem = list(logprob_entry.values())[0]
        token = logprobItem.decoded_token.strip()
        prob = math.exp(logprobItem.logprob)
        # rank = logprobItem.rank
        if "(" in token:
            parenthesis_flag = True
        if parenthesis_flag and token.strip("()") in ["A", "B", "C", "D"]:
            print(prob, token)
            return prob, token
    return 0, ""


class VLlmClient(OpenAIVllmClient):
    def __init__(
        self,
        cfg_vllm: DictConfig | None = None,
    ):
        if cfg_vllm is None:
            self.llm = LLM(
                model="OpenGVLab/InternVL2_5-1B-MPO",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                trust_remote_code=True,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 8},
            )
            self.name = "OpenGVLab/InternVL2_5-1B-MPO"
            self.max_model_len = 4096
        else:
            self.llm = LLM(
                model=cfg_vllm.model_name,
                tensor_parallel_size=cfg_vllm.tensor_parallel_size,
                gpu_memory_utilization=cfg_vllm.gpu_memory_utilization,
                enforce_eager=cfg_vllm.enforce_eager,
                trust_remote_code=True,
                max_model_len=cfg_vllm.max_tokens,
                limit_mm_per_prompt={"image": cfg_vllm.images_per_prompt},
            )
            self.name = cfg_vllm.model_name
            self.max_model_len = cfg_vllm.max_tokens

    def chat_response(
        self,
        user_content,
        max_tokens,
        temperature=0.0,
        logprobs=True,  # unused
        top_logprobs=-1,
        top_k=-1,
        top_p=1,
        seed=0,
    ):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=top_logprobs,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        outputs = self.llm.chat(
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ],
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0]

    def prompt_video(
        self,
        prompt,
        path: str | Path | list[Image.Image],
        frame_interval=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs: int = 1,
        top_k: int = -1,
        top_p: int = 1,
        seed: int = 0,
        crop: bool = False,
    ):
        # Append the CoT prompt to the user prompt
        prompt = prompt + self.COT_PROMPT
        user_content = [{"type": "text", "text": prompt}]
        random.seed(seed)
        np.random.seed(seed)

        # path is a list of PIL images
        if isinstance(path, list):
            frame_count = 0
            for i, frame in enumerate(path):
                if isinstance(frame, Image.Image):
                    frame_count += 1
                    if crop:
                        cropped_images = crop_image(frame)
                        # Randomly select one cropped image
                        frame = random.choice(cropped_images)
                    else:
                        frame = np.array(frame)
                    encoded_frame = self._encode_frame(frame)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )
                if frame_count >= 100:
                    print("Can only process 100 frames")
                    break
        # Check if the video file is an MP4
        elif path.endswith(".mp4"):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            # Initialize user content with the prompt text
            frame_count = 0
            # Read frames from the video
            while cap.isOpened():
                ret, frame = cap.read()  # frame is a numpy array
                if not ret:
                    break
                if crop:
                    cropped_images = crop_image(frame)
                    # Randomly select one cropped image
                    frame = random.choice(cropped_images)

                # Process every nth frame based on frame_interval
                if frame_count % frame_interval == 0:
                    encoded_frame = self._encode_frame(frame)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )

                frame_count += 1
                if frame_count >= 100:
                    print("Can only process 100 frames")
                    break
            cap.release()
        # Check if the video file is a GIF
        elif path.endswith(".gif"):
            gif = imageio.mimread(path)
            frame_count = 0

            # Process every nth frame based on frame_interval
            for i, frame in enumerate(gif):
                frame_count += 1
                if i % frame_interval == 0:
                    # Convert to RGB explicitly using PIL
                    frame_image = Image.fromarray(frame)
                    frame_image = frame_image.convert("RGB")
                    if crop:
                        cropped_images = crop_image(frame)
                        # Randomly select one cropped image
                        frame = random.choice(cropped_images)

                    # Convert PIL image to OpenCV format (RGB to BGR)
                    # This step is necessary because _encode_frame uses cv2.imencode, which expects BGR format
                    cv_image = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)

                    encoded_frame = self._encode_frame(cv_image)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )
                if frame_count >= 100:
                    print("Can only process 100 frames")
                    break
        # if the path is a directory
        elif os.path.isdir(path):
            frame_count = 0
            for file in sorted(os.listdir(path)):
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    frame_count += 1
                    if crop:
                        cropped_images = crop_image(os.path.join(path, file))
                        # Randomly select one cropped image
                        frame = random.choice(cropped_images)
                    else:
                        frame = os.path.join(path, file)
                    encoded_frame = self._encode_frame(frame)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )
                if frame_count >= 100:
                    print("Can only process 100 frames")
                    break
        # Raise an error if the video format is unsupported
        else:
            raise ValueError("Unsupported video format")
        print(f"Processed {frame_count} frames with interval {frame_interval}")
        return self.chat_response(
            user_content, self.max_model_len, temperature, logprobs, top_logprobs, top_k, top_p, seed
        )

    # unused
    # def format_prompt(self, user_prompt, num_images):
    #     prompt = self.SYSTEM_MESSAGE + "<|user|>\n"
    #     if num_images > 0:
    #         prompt += "".join([f"Frame{i + 1}: <image>" for i in range(num_images)])
    #     prompt += self.COT_PROMPT
    #     prompt += user_prompt
    #     prompt += "<|assistant|>:\n"
    #     return prompt

    # def prompt_video(self, prompt, path, frame_interval=1):
    # # query with generate template
    # images = []
    # if path.endswith(".mp4"):
    #     cap = cv2.VideoCapture(path)
    #     if not cap.isOpened():
    #         raise ValueError("Could not open video")

    #     frame_count = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if frame_count % frame_interval == 0:
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             images.append(image)
    #         frame_count += 1
    #     cap.release()
    # elif path.endswith(".gif"):
    #     gif = imageio.mimread(path)
    #     for i, frame in enumerate(gif):
    #         if i % frame_interval == 0:
    #             frame_image = Image.fromarray(frame).convert("RGB")
    #             images.append(frame_image)
    # elif os.path.isdir(path):
    #     for file in sorted(os.listdir(path)):
    #         if file.endswith(('.jpg', '.jpeg', '.png')):
    #             image = Image.open(os.path.join(path, file))
    #             images.append(image)
    # else:
    #     raise ValueError("Unsupported video format")

    # prompt = self.format_prompt(prompt, len(images))
    # print(prompt)
    # outputs = self.llm.generate({"prompt": prompt, "multi_modal_data": {"image": images}}, self.sampling_params)


if __name__ == "__main__":
    vllm = VLlmClient()
    response = vllm.prompt_video(
        prompt="Where was the video taken? Select one from the following options: (A) Others, (B) Mountainous area,\
                (C) Forest, (D) A straight road",
        # path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        path="/nas/pohan/datasets/SUTDTrafficQA/raw_videos/b_1a4411B7sb_clip_005.mp4",
        crop=True,
    )
    print(response)
    # SamplingParams(temperature=0.2, max_tokens=MAX_TOKENS, logprobs=1, top_k=5, seed = 16)
