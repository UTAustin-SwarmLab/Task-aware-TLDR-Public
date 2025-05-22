# CUDA_VISIBLE_DEVICES=1 python -m src.utils.fill_in_mask
import os
import re
import sys
from pathlib import Path

import tiktoken
from PIL import Image
from transformers import AutoTokenizer

# Fix the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.client.vllm_client import VLlmClient


def TTTokenizeText(model_name: str, text: str) -> list[str]:
    # Load the tokenizer
    tokenizer = tiktoken.get_encoding(model_name)
    # Tokenize the input
    token_ids = tokenizer.encode(text)
    # Decode each token
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return tokens


def HFTokenizeText(model_name, text: str, tokenizer: AutoTokenizer | None = None) -> list[str]:
    if tokenizer is None:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Check if the tokenizer is a Tiktoken tokenizer
    elif hasattr(tokenizer, "tiktoken_encoding"):
        # Use Tiktoken tokenizer
        return TTTokenizeText(tokenizer.tiktoken_encoding, text)

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")
    # Get the token IDs
    token_ids = inputs["input_ids"][0].tolist()
    # Decode each token
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return tokens


def IsCorrectPrediction(prediction, ground_truth, tokenizer=None):
    """
    Determine if a prediction is correct, allowing for token completion.

    Args:
        prediction: The model's prediction (could be a subword)
        ground_truth: The expected complete word
        tokenizer: Optional tokenizer to check tokenization (model name)

    Returns:
        Boolean indicating if prediction should be considered correct
    """
    # Normalize case
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    # Case 1: Exact match
    if prediction == ground_truth:
        return True

    # Case 2: Significant prefix match (at least 3 chars or 50% of word)
    min_prefix_length = min(3, len(ground_truth) // 2)
    if ground_truth.startswith(prediction) and len(prediction) >= min_prefix_length:
        return True

    # Case 3: First token match (if tokenizer provided)
    if tokenizer:
        tokens = HFTokenizeText(tokenizer, ground_truth)
        if tokens:
            first_token = tokens[0].replace("Ġ", "").lower()
            if prediction == first_token:
                return True

    return False


def IsCorrectChoice(prediction, ground_truth):
    """
    Determine if a prediction is the correct choice in (A), (B), (C), (D).

    Args:
        prediction: The model's prediction (could be a subword)
        ground_truth: The expected complete word

    Returns:
        Boolean indicating if prediction should be considered correct
    """
    # Normalize case and remove parentheses
    prediction = prediction.strip().lower().replace("(", "").replace(")", "")
    ground_truth = ground_truth.strip().lower().replace("(", "").replace(")", "")

    # Case 1: Exact match
    return prediction == ground_truth


def MaskedVideo2LogProb(
    orig_prompt: str,
    answer_choice: list[str],
    video_path: str | Path | list[Image.Image],
    vllm_model: str | VLlmClient = "OpenGVLab/InternVL2_5-1B-MPO",
    temperature: float = 0.0,
    top_logprobs: int = 20,
    min_logprob: float = -5.29831736655,
    crop: bool = False,
) -> list[tuple[str, float, float]]:
    """
    Fill in the masked words in the text using the VLLM model. Note that the masked words are replaced with "<MASK>"
    in the prompt. The function will prompt the user to fill in the masked words one by one and return the predicted
    masked words and their **LOG** probabilities.

    Args:
        orig_prompt (str): The original text prompt.
        answer_choice (str): The correct answer choice, e.g. B.
        video_path (str): The path to the video file. Default is None.
        vllm_model (str): The VLLM model name. Default is "OpenGVLab/InternVL2_5-1B-MPO".
        temperature (float): The temperature for sampling. Default is 0.0.
        top_logprobs (int): The number of top log probabilities to return. Default is 1.
        min_logprob (float): The minimum log probability. Default is -5.29831736655 (0.5%).
        crop (bool): Whether to crop the video. Default is False.

    Returns:
        list[tuple[str, float, float]]: The predicted answer, log probability, and rank.
    """
    video_path = str(video_path) if not isinstance(video_path, list) else video_path
    instruction = """Analyze the video where certain content is black-masked and outlined with red boundaries.
    Use the visible information to answer the following question:\n"""

    # Create a VLLM client
    if isinstance(vllm_model, str):
        vllm = VLlmClient()
    else:
        vllm = vllm_model
    vllm.SYSTEM_MESSAGE = ""
    vllm.COT_PROMPT = ""

    response = vllm.prompt_video(
        instruction + orig_prompt,
        video_path,
        frame_interval=1,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
        crop=crop,
    )
    pred_hit = False
    for logprobItem in list(response.logprobs[0].values()):
        pred_token = logprobItem.decoded_token
        log_prob = logprobItem.logprob
        print("Token", pred_token, "Logprob", log_prob, "Rank", logprobItem.rank, "Answer choice", answer_choice)
        if IsCorrectChoice(pred_token, answer_choice):
            tok_prob_rank = (pred_token, log_prob, logprobItem.rank)
            pred_hit = True
            print("Correct!!!!")
            break
        elif log_prob <= min_logprob:
            break
    if not pred_hit:  # If the correct answer is not found in the top logprobs
        tok_prob_rank = (answer_choice, min_logprob, -1)
        print("No match!!!!", answer_choice, log_prob)
    return tok_prob_rank


def FillMaskText2LogProb(
    orig_prompt: str,
    masked_words_set: set[str],
    video_path: str | Path | Image.Image | None = None,
    vllm_model: None | VLlmClient = None,
    temperature: float = 0.0,
    top_logprobs: int = 20,
    min_logprob: float = -5.29831736655,
) -> list[tuple[str, float, float]]:
    """
    Fill in the masked words in the text using the VLLM model. Note that the masked words are replaced with "<MASK>"
    in the prompt. The function will prompt the user to fill in the masked words one by one and return the predicted
    masked words and their **LOG** probabilities.

    Args:
        orig_prompt (str): The original text prompt.
        masked_tokens_set (set[str]): The set of masked tokens to be filled in.
        video_path (str): The path to the video file. Default is None.
        vllm_model (None | VLlmClient): The VLLM model. Default is None.
        temperature (float): The temperature for sampling. Default is 0.0.
        top_logprobs (int): The number of top log probabilities to return. Default is 1.
        min_logprob (float): The minimum log probability. Default is -5.29831736655 (0.5%).

    Returns:
        list[tuple[str, float, float]]: The predicted masked words, their log probabilities, and ranks.
    """
    tok_prob_rank = []
    instruction = """Guess and fill in the first <MASK> in the sentence. For example, given: I <MASK> <MASK> dog,
    the correct answer would be 'am,' interpreting the sentence as 'I am a dog.'
    Respond only with the word for the first masked word, without any explanation. Here is the text to complete:"""
    print(instruction)

    prompt_words = re.findall(r"\b\w+\b", orig_prompt)
    print("Prompt Tokens", prompt_words)
    print("Masked Tokens", masked_words_set)

    # Store masked_words in the order they appear in orig_prompt, accounting for plural and capital cases
    masked_words_in_order = []
    for word in prompt_words:
        for masked_word in masked_words_set:
            if word.lower() == masked_word.lower() or word.lower() == masked_word.lower() + "s":
                masked_words_in_order.append(masked_word)
                break
    print("Masked tokens in order", masked_words_in_order)
    masked_prompt = orig_prompt
    for word in prompt_words:
        if word in masked_words_set:
            # get what is the match in the masked_tokens
            masked_prompt = re.sub(rf"\b{re.escape(word)}\b", "<MASK>", masked_prompt, count=1, flags=re.IGNORECASE)

    # Create a VLLM client
    if vllm_model is None:
        vllm = VLlmClient()
    else:
        vllm = vllm_model
    vllm_model_name = vllm.name
    vllm.SYSTEM_MESSAGE = ""
    vllm.COT_PROMPT = ""

    for masked_word in masked_words_in_order:
        print("Current Prompt: ", masked_prompt)
        if video_path is None:
            response = vllm.prompt(
                instruction + masked_prompt,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
        elif isinstance(video_path, Path):
            video_path = str(video_path)
        if isinstance(video_path, list) or isinstance(video_path, str):
            response = vllm.prompt_video(
                instruction + masked_prompt,
                video_path,
                frame_interval=1,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
            )

        pred_hit = False
        for logprobItem in list(response.logprobs[0].values()):
            pred_token = logprobItem.decoded_token
            log_prob = logprobItem.logprob
            rank = logprobItem.rank
            # print("Token", pred_token, "Logprob", log_prob, "Rank", logprobItem.rank, "Masked word", masked_word)
            if IsCorrectPrediction(pred_token, masked_word, tokenizer=vllm_model_name):
                tok_prob_rank.append((pred_token, log_prob, rank, masked_word))
                pred_hit = True
                print("Token", pred_token, "Logprob", log_prob, "Rank", logprobItem.rank, "Masked word", masked_word)
                print("Correct!!!!")
                break
            elif log_prob <= min_logprob:
                break
        if not pred_hit:  # If the correct answer is not found in the top logprobs
            tok_prob_rank.append((masked_word, min_logprob, -1, masked_word))
            print("No match!!!!", masked_word, min_logprob)

        masked_prompt = masked_prompt.replace("<MASK>", masked_word, 1)
    assert len(tok_prob_rank) == len(masked_words_in_order), (
        "Mismatch in the number of masked words and predictions, {} vs {}".format(
            len(masked_words_in_order), len(tok_prob_rank)
        )
    )
    return tok_prob_rank


if __name__ == "__main__":
    # orig_prompt = "I ride a bike to school on a Monday morning."
    # masked_tokens = [" ride", " a", " morning"]

    tldr = "The presentation discusses the use of graph-based models to analyze data"
    MaskedVideo2LogProb(
        # """Analyze the slides from a paper presentation and identify the relevant research area.
        # Respond only with the corresponding letter:
        # (A) Self-supervised Representation Learning (B) Graph Neural Network (C) Computer Vision (D) Optimization.""",
        f"""Analyze the slides from a paper presentation and the provided summary to identify the relevant research area.
        Summary: {tldr}.
        Respond only with the corresponding letter:
        (A) Self-supervised Representation Learning (B) Graph Neural Network (C) Computer Vision (D) Optimization.""",
        "B",
        "/home/ta-tldr/Project/tldr/slide_test/",
        "OpenGVLab/InternVL2_5-1B-MPO",
    )
