"""
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://localhost:8000/v1 -v open-webui:/app/backend/data \
    --name open-webui --restart always ghcr.io/open-webui/open-webui:main
docker stop open-webui && docker rm open-webui
"""

import base64
import io
import math
import os

import cv2
import imageio
import numpy as np
from openai import OpenAI
from PIL import Image


def parse_selection_token_prob(response_choice):
    parenthesis_flag = False
    for content in response_choice.logprobs.content:
        # print(content.token.strip(), content.logprob)
        if "(" in content.token.strip():
            parenthesis_flag = True
        if parenthesis_flag and content.token.strip().strip("(").strip(")") in ["A", "B", "C", "D"]:
            prob = math.exp(content.logprob)
            answer = content.token.strip()
            print(prob, answer)
            return prob, answer
    return 0, ""


class OpenAIVllmClient:
    # MODEL = "OpenGVLab/InternVL2_5-38B-MPO"
    # MODEL = "OpenGVLab/InternVL2_5-78B-AWQ"
    # MODEL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    MODEL = "OpenGVLab/InternVL3-38B"
    SYSTEM_MESSAGE = "You are a visual analysis assistant that carefully examines images, given in chronological."
    COT_PROMPT = """
    To answer the previous prompt, please follow the CoT format:
    1. Analyze what you see with <Think>: Provide detailed and factual observations and reasoning step-by-step.
    2. Summarize your analysis with <TL;DR>: Concisely capture all key points from your thinking.
    3. Answer the question directly with <Answer>: Use a ( and ) with an alphabet in between.
    """
    MAX_TOKENS = 1024

    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model=MODEL,
        MAX_TOKENS=1024,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.system_message = self.SYSTEM_MESSAGE
        self.MAX_TOKENS = MAX_TOKENS

    def _encode_frame(self, frame):
        # Resize the frame to 512x512 pixels
        if isinstance(frame, np.ndarray):
            frame = cv2.resize(frame, (512, 512))
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                raise ValueError("Could not encode frame")
            return base64.b64encode(buffer).decode("utf-8")
        elif isinstance(frame, str):
            image = Image.open(frame)
            image = image.resize((512, 512))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError("Unsupported frame type. Expected numpy array or file path.")

    def chat_response(
        self,
        user_content,
        max_tokens,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
        top_p=None,
        seed=None,
    ):
        # Create a chat response using the user content
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            top_p=None,
            seed=None,
        )

        return chat_response.choices[0]

    def prompt(
        self,
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
    ):
        prompt = prompt + self.COT_PROMPT

        # Build the user message: a text prompt plus one image for each frame.
        user_content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        return self.chat_response(user_content, max_tokens, temperature, logprobs, top_logprobs)

    def prompt_image(
        self,
        prompt,
        path: np.ndarray | str,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
    ):
        # Encode each frame.
        encoded_image = self._encode_frame(path)
        prompt = prompt + self.COT_PROMPT

        # Build the user message: a text prompt plus one path for each frame.
        user_content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:path/jpeg;base64,{encoded_image}"},
            }
        )
        return self.chat_response(user_content, max_tokens, temperature, logprobs, top_logprobs)

    def prompt_video(
        self,
        prompt,
        path,
        frame_interval=1,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
    ):
        # Append the CoT prompt to the user prompt
        prompt = prompt + self.COT_PROMPT
        user_content = [{"type": "text", "text": prompt}]

        # Check if the video file is an MP4
        if path.endswith(".mp4"):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            # Initialize user content with the prompt text
            frame_count = 0
            # Read frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

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
            # Uniformly sample 20 frames
            if frame_count >= 20:  # Exclude the initial text prompt
                print("Can only process 20 frames")
                step = max(1, len(user_content[1:]) // 20)
                user_content = user_content[:1] + user_content[1::step][:20]
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

                    # Convert PIL image to OpenCV format (RGB to BGR)
                    # This step is necessary because _encode_frame uses cv2.imencode
                    # which expects BGR format
                    cv_image = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)

                    encoded_frame = self._encode_frame(cv_image)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )
            if frame_count >= 20:
                print("Can only process 20 frames")
                step = max(1, len(user_content[1:]) // 20)
                user_content = user_content[:1] + user_content[1::step][:20]
        # if the path is a directory
        elif os.path.isdir(path):
            frame_count = 0
            for file in sorted(os.listdir(path)):
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    frame_count += 1
                    encoded_frame = self._encode_frame(os.path.join(path, file))
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"},
                        }
                    )
            if frame_count >= 20:
                print("Can only process 20 frames")
                step = max(1, len(user_content[1:]) // 20)
                user_content = user_content[:1] + user_content[1::step][:20]

        # Raise an error if the video format is unsupported
        else:
            raise ValueError("Unsupported video format")
        print(f"Processed {frame_count} frames with interval {frame_interval}")
        return self.chat_response(user_content, max_tokens, temperature, logprobs, top_logprobs)


if __name__ == "__main__":
    vllm = OpenAIVllmClient()
    # response_choice = vllm.prompt("How many 'r's are there in 'strawberry'?", temperature=0.2)
    # print(response_choice.message.content)
    # exit()

    # response_choice = vllm.prompt("Hello, how are you?", temperature=0)
    # print(response_choice.message.content)
    # exit()

    # print(
    #     vllm.prompt_image(
    #         prompt="Subtract all tiny shiny balls. Subtract all purple objects. How many objects are left? \
    #           Tell me your thoughts step-by-step.",
    #         path="/home/ta-tldr/Project/tldr/asset/CoT_reasoning.png",
    #     ).message.content,
    #     "\n--------------------------------\n",
    # )
    # print(
    #     vllm.prompt_video(
    #         prompt="What do you see in the video?", path="/home/ta-tldr/Project/tldr/asset/7.gif"
    #     ).message.content,
    #     "\n--------------------------------\n",
    # )

    # print(
    #     vllm.prompt_video(
    #         prompt="What does the model do after lowering her coat?",
    #         path="/home/ta-tldr/Project/tldr/asset/model.gif",
    #         frame_interval=5,
    #     ).message.content
    # )
    # print(
    #     vllm.prompt_video(
    #         prompt="What is the color of the bulldog?", path="/home/ta-tldr/Project/tldr/asset/dog.gif"
    #     ).message.content
    # )

    vllm.SYSTEM_MESSAGE = ""
    vllm.COT_PROMPT = ""
    print(
        vllm.prompt_video(
            # prompt="Summarize what you see from the video in the chronological order.",
            prompt="Describe the accident of the vehicle in the video in 4 sentences.",
            path="/nas/pohan/datasets/SUTDTrafficQA/raw_videos/b_1a4411e7sC_clip_033.mp4",
        ).message.content
    )
    exit()

    # Q1–
    print(
        vllm.prompt_video(
            prompt="Where was the video taken? Select one from the following options: (A) Others, (B) Mountainous area,\
                (C) Forest, (D) A straight road",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q2
    print(
        vllm.prompt_video(
            prompt="Which factors might have contributed to the accident? Select one from the following options: \
                (A) There is no accident in this video, (B) Speeding vehicles, \
                (C) Improper lane change, (D) Traffic light violation",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q3
    print(
        vllm.prompt_video(
            prompt="Where was the video taken? Select one from the following options: (A) Expressway, \
                (B) A turning point, (C) Tunnel, (D) Road in the city",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q4
    print(
        vllm.prompt_video(
            prompt="Which factors might have contributed to the accident? Select one from the following options: \
                (A) Traffic congestion, (B) Bad road surfaces, (C) Others, (D) Fatigue driving",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q5
    print(
        vllm.prompt_video(
            prompt="How did the truck get involved in the accident? Select one from the following options: \
                (A) The truck is being hit from behind. (B) The truck is being hit from the side.\
                (C) The truck is not involved in the accident.",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q6
    print(
        vllm.prompt_video(
            prompt="Which factors might have contributed to the accident? Select one from the following options:\
                (A) Retrograde vehicles, (B) Obstructed view, (C) Improper lane change,\
                (D) Not paying attention to their surroundings or road safety",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )
    # Q7
    print(
        vllm.prompt_video(
            prompt="What's the time of the day? Select one from the following options: (A) Nighttime, (B) Daytime",
            path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
        ).message.content
    )

    # print(
    #     vllm.prompt_video(
    #         prompt="Summarize what you see from the video in the chronological order. Then, answeer the question: \
    #             What's the time of the day? Possibele answers: Nighttime, Daytime",
    #         path="/home/ta-tldr/Project/tldr/keyframes/b_1a4411B7sb_clip_005/",
    #     ).message.content
    # )
    # print(
    #     vllm.prompt_video(
    #         prompt="What does the cat do 3 times? Tell me step-by-step.",
    #         path="/home/ta-tldr/Project/tldr/asset/3.gif",
    #         frame_interval=5,
    #     ).message.content
    # )
    # print(
    #     vllm.prompt_video(
    #         prompt="How many times does the cat lick the water in the video? Tell me your thoughts step-by-step.",
    #         frame_interval=5,
    #         path="/home/ta-tldr/Project/tldr/asset/7.gif",
    #     ).message.content
    # )
    # print(
    #     vllm.prompt_video(
    #         prompt="Summarize what the cat did in chronological order. Make it as detailed as possible.",
    #         frame_interval=5,
    #         path="/home/ta-tldr/Project/tldr/asset/7.gif",
    #     ).message.content
    # )
