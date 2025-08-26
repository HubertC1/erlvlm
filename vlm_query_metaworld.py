import os
import time
import enum
import json

import numpy as np
# import google.generativeai as genai
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from google.generativeai import GenerationConfig
# from google.generativeai.types.generation_types import BrokenResponseError

from utils import create_image_from_numpy
from gemini_keys import gemini_api_keys
# from qwen2_vl_interface import LVLMModel
from llava.llava_interface import initialize_llava_model, generate_text_from_image
import cv2


SLEEP_AFTER_TRY = 5  # seconds
DEFAULT_SLEEP = 5

gemini_api_keys_for_mp = []
global_key_idx_for_mp = []


def construct_gemini_keys(num_rollout_workers):
    # For LLaVA, we don't need API keys but keep the interface for compatibility
    global gemini_api_keys_for_mp
    global global_key_idx_for_mp

    # Initialize LLaVA model once
    cache_dir = "/tmp/llava_model"
    initialize_llava_model(cache_dir)
    print("LLaVA model initialized for VLM queries")

    # For compatibility, set up dummy structures
    gemini_api_keys_for_mp = [["llava"] for _ in range(num_rollout_workers)]
    global_key_idx_for_mp = [0] * num_rollout_workers


def construct_vlm_with_key(rank=0, model_name=None):
    # Set up the model parameters: https://ai.google.dev/gemini-api/docs/models/generative-models#model_parameters
    # https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pro-config-example

    global gemini_api_keys_for_mp
    global global_key_idx_for_mp
    assert len(global_key_idx_for_mp) > 0, "List of key is empty, do you run construct_gemini_keys()?"
    # For LLaVA, use dummy key management
    gemini_key = gemini_api_keys_for_mp[rank][global_key_idx_for_mp[rank]]
    
    if model_name is None:
        model_name = "llava"
    
    # Create a simple LLaVA model class
    class LLaVAModel:
        def __init__(self):
            self.name = "llava"
    
    vlm_model = LLaVAModel()

    global_key_idx_for_mp[rank] += 1
    if global_key_idx_for_mp[rank] == len(gemini_api_keys_for_mp[rank]):
        global_key_idx_for_mp[rank] = 0
    return vlm_model, gemini_key


def handling_query_error(error, attempt_cnt, rank, gemini_key):
    if hasattr(error, 'grpc_status_code'):
        if error.grpc_status_code.name in ["DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL", "UNKNOWN", "UNAVAILABLE"]:
            # Reference error: https://cloud.google.com/apis/design/errors#handling_errors
            print(f"[INFO] Got '{error.grpc_status_code.name}' from Gemini, retrying ({attempt_cnt}) ... (rank={rank}, {gemini_key})")
            time.sleep(SLEEP_AFTER_TRY + (attempt_cnt - 1) * 3)

        else:
            print(f"[INFO] Unknown grpc_status_code: {error}, re-trying ({attempt_cnt}) ... (rank={rank}, {gemini_key})")
            time.sleep(SLEEP_AFTER_TRY + (attempt_cnt - 1) * 3)
    else:
        print(f"[INFO] Unknown error {error}, re-trying ({attempt_cnt}) ... (rank={rank}, {gemini_key})")
        time.sleep(SLEEP_AFTER_TRY + (attempt_cnt - 1) * 3)


def query_vlm_for_summarization(vlm_model, image, given_instruction, sleep=DEFAULT_SLEEP):
    prompt_analyze = (
        f"You will be presented an image of a robot arm performing the task [{given_instruction}]. "
        f"Please focus on the target object in the task and carefully analyze the image in term of **completing** the task. "
    )

    error = None
    gemini_text = []
    # chat = vlm_model.start_chat()
    # try:
    #     responses = chat.send_message([image, prompt_analyze], stream=True)
    #     responses.resolve()
    # except Exception as error:
    #     return None, [], error

    try:
        # Use LLaVA interface
        vlm_analysis = generate_text_from_image(image, prompt_analyze)
    except Exception as error:
        return None, [], error

    vlm_analysis = vlm_analysis.strip()
    gemini_text.append(prompt_analyze)
    gemini_text.append(vlm_analysis)
    time.sleep(sleep)
    return vlm_analysis, gemini_text, error


def query_vlm_for_rating_given_summarization(vlm_model, trajectory_analysis, given_instruction, max_rating):
    n_rating_classes = max_rating + 1
    class Choice(enum.Enum):
        if n_rating_classes == 2:
            RATING_0 = "Bad"
            RATING_1 = "Good"
        elif n_rating_classes == 3:
            RATING_0 = "Bad"
            RATING_1 = "Average"
            RATING_2 = "Good"
        elif n_rating_classes == 4:
            RATING_0 = "Very Bad"
            RATING_1 = "Bad"
            RATING_2 = "Good"
            RATING_3 = "Very Good"
        elif n_rating_classes == 5:
            RATING_0 = "Very Bad"
            RATING_1 = "Bad"
            RATING_2 = "Average"
            RATING_3 = "Good"
            RATING_4 = "Very Good"

        @staticmethod
        def get_value(idx):
            return Choice.__members__[f"RATING_{idx}"].value

    rating_category = "[" + "".join(f"{Choice.get_value(i)}, " for i in range(n_rating_classes - 1)) + f"{Choice.get_value(n_rating_classes - 1)}]"
    prompt_conclusion = (
        f"\n{trajectory_analysis}\n"
        f"From the above analyses, based on this rating category: {rating_category}, "
        f"how would you rate this image in terms of **completing** task [{given_instruction}]?"
    )
    error = None
    gemini_text = []
    try:
        # Use LLaVA interface for rating - create a simple text prompt
        rating_prompt = f"{prompt_conclusion} Please respond with only one of these options: {rating_category}"
        gemini_analysis = generate_text_from_image(None, rating_prompt)  # Text-only prompt for rating
    except Exception as error:
        return None, [], error

    # Since LLaVA gives free-form text, we need to parse the rating
    gemini_analysis = gemini_analysis.strip()
    gemini_text.append(prompt_conclusion)
    gemini_text.append(gemini_analysis)

    final_reward_answer = None
    if n_rating_classes == 2:
        if gemini_analysis == Choice.RATING_0.value:
            final_reward_answer = 0
        elif gemini_analysis == Choice.RATING_1.value:
            final_reward_answer = 1

    elif n_rating_classes == 3:
        if gemini_analysis == Choice.RATING_0.value:
            final_reward_answer = 0
        elif gemini_analysis == Choice.RATING_1.value:
            final_reward_answer = 1
        elif gemini_analysis == Choice.RATING_2.value:
            final_reward_answer = 2

    elif n_rating_classes == 4:
        if gemini_analysis == Choice.RATING_0.value:
            final_reward_answer = 0
        elif gemini_analysis == Choice.RATING_1.value:
            final_reward_answer = 1
        elif gemini_analysis == Choice.RATING_2.value:
            final_reward_answer = 2
        elif gemini_analysis == Choice.RATING_3.value:
            final_reward_answer = 3

    elif n_rating_classes == 5:
        if gemini_analysis == Choice.RATING_0.value:
            final_reward_answer = 0
        elif gemini_analysis == Choice.RATING_1.value:
            final_reward_answer = 1
        elif gemini_analysis == Choice.RATING_2.value:
            final_reward_answer = 2
        elif gemini_analysis == Choice.RATING_3.value:
            final_reward_answer = 3
        elif gemini_analysis == Choice.RATING_4.value:
            final_reward_answer = 4

    else:
        raise NotImplementedError

    assert final_reward_answer is not None, f"gemini_analysis={gemini_analysis}"
    return final_reward_answer, gemini_text, error


def vlm_reasoning_rating_metaworld(
        raw_observations,
        model_names=(None, None),
        env_name=None,
        max_rating=5,
        rank=0,
        sleep=DEFAULT_SLEEP,
        default_window_size=1,
):
    assert len(model_names) == 2, f"Confused selected models: {model_names}"
    model_names = tuple(model_names)
    trajectory_len = raw_observations.shape[0]
    given_instruction = preprocess_instruction(env_name)

    MAX_ATTEMPT = 5  # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/api-errors#handle-errors

    whole_conversation = []
    trajectory_vlm_analysis, images_to_ask_vlm = [], []
    vlm_model, error_duing_query_vlm = None, False
    start, total_times_query, total_times_query_success = 0, 0, 0
    """================================ Start summarize using VLM ================================"""
    while (start < trajectory_len and not error_duing_query_vlm):
        end = start + default_window_size
        seq_1 = raw_observations[start:end]
        image_to_ask_vlm = create_image_from_numpy(seq_1[0])

        attempt_cnt, query_success = 0, False
        while not query_success:
            vlm_model, gemini_key = construct_vlm_with_key(rank=rank, model_name=model_names[0])
            if True and rank in [0, 1]:
                cv2.imshow(f"image_to_ask_vlm_rank{rank}", cv2.cvtColor(np.array(image_to_ask_vlm), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
            vlm_analysis, gemini_text, error = query_vlm_for_summarization(
                vlm_model=vlm_model,
                image=image_to_ask_vlm,
                given_instruction=given_instruction,
                sleep=sleep,
            )
            total_times_query += 1
            attempt_cnt += 1

            if error is None:
                query_success = True
                total_times_query_success += 1
                trajectory_vlm_analysis.append(vlm_analysis)
                whole_conversation.append(gemini_text)
                images_to_ask_vlm.append(image_to_ask_vlm)

            else:
                if attempt_cnt == MAX_ATTEMPT:
                    error_duing_query_vlm = True
                    print(f"[INFO] Ignore querying Gemini after tried {MAX_ATTEMPT}.")
                    break

                # For LLaVA, we don't have BrokenResponseError, so just retry with sleep
                print(f"[INFO] Got error during LLaVA query, re-trying ({attempt_cnt}) ... (rank={rank})")
                time.sleep(SLEEP_AFTER_TRY + (attempt_cnt - 1) * 3)

        start += default_window_size

    if error_duing_query_vlm:
        query_info = dict(
            total_times_query=total_times_query,
            total_times_query_success=total_times_query_success,
            gemini_texts=whole_conversation,
            images_to_ask_vlm=images_to_ask_vlm,
            error_during_query=True,
        )
        return 0.0, query_info

    vlm_summary = "".join("{}".format(text) for text in trajectory_vlm_analysis)

    error_duing_query_vlm = False
    attempt_cnt, query_success = 0, False
    final_reward_answer = 0.0
    """================================ Get rating from LLM ================================"""
    while not query_success:
        vlm_model, gemini_key = construct_vlm_with_key(rank=rank, model_name=model_names[1])
        vlm_reward, gemini_text, error = query_vlm_for_rating_given_summarization(
            vlm_model=vlm_model,
            trajectory_analysis=vlm_summary,
            given_instruction=given_instruction,
            max_rating=max_rating
        )
        total_times_query += 1
        attempt_cnt += 1

        if error is None:
            final_reward_answer = vlm_reward
            query_success = True
            total_times_query_success += 1
            whole_conversation.append(gemini_text)

        else:
            if attempt_cnt == MAX_ATTEMPT:
                error_duing_query_vlm = True
                print(f"[INFO] Ignore querying Gemini after tried {MAX_ATTEMPT}.")
                break

            elif isinstance(error, BrokenResponseError):
                print(f"[INFO] Got BrokenResponseError, re-trying ({attempt_cnt}) ... (rank={rank}, {gemini_key})")
                time.sleep(SLEEP_AFTER_TRY + (attempt_cnt - 1) * 3)

            else:
                handling_query_error(error, attempt_cnt, rank, gemini_key)

    query_info = dict(
        total_times_query=total_times_query,
        total_times_query_success=total_times_query_success,
        gemini_texts=whole_conversation,
        images_to_ask_vlm=images_to_ask_vlm,
        error_during_query=error_duing_query_vlm
    )
    return final_reward_answer, query_info



def preprocess_instruction(env_name):
    all_instructions = {
        "metaworld_sweep-into-v2": "place the green cube so that it lies on the square hole",
        "metaworld_drawer-open-v2": "open the drawer",
        "metaworld_soccer-v2": "place the soccer ball so that it lies inside the goal",
    }
    instruction = all_instructions[env_name]
    return instruction

