import os
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import re
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from vllm import LLM, SamplingParams
from datasets import load_dataset
from sal.config import Config
from sal.models.reward_models import PRM
from sal.models.reward_models import RLHFFlow
from sal.utils.score import aggregate_scores



#############################################################################################################
############################################## Oracle Verifier ##############################################
#############################################################################################################
def best_of_n_oracle(dataset, config: Config, llm: LLM, N=Config.n, save_results=False):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""
        true_answer = data['answer']
        x = {"question": [data['question']]}

        convs = [
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in x["question"]
        ]
        # TODO: set the augmented template from a file
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        # Duplicate convs to generate config.n completions per prompt so we can do continous batching
        # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
        templated_convs = [c for conv in templated_convs for c in [conv] * N]

        # Initialize empty lists for completions and completion tokens
        completions = [[] for _ in range(len(x["question"]))]
        completion_tokens = [[] for _ in range(len(x["question"]))]

        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(responses) != len(x["question"]) * N:
            raise ValueError(
                f"Generated {len(responses)} responses instead of {len(x['question'] * N)}"
            )

        for i in range(len(completions)):
            completions[i] = [
                output.text
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]
            completion_tokens[i] = [
                len(output.token_ids)
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]

        # Check we generated the correct number of completions for each prompt
        for c in completions:
            if len(c) != N:
                raise ValueError(f"Generated {len(c)} completions instead of {N}")


        


        oracle_response = ""
        response = completions[0][0]

        response = response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1)

        for completion in completions[0]:
            match = re.search(r'\$\\boxed{(.*?)}\$', completion)
            if match:
                oracle_response = match.group(1)
                if oracle_response == true_answer:
                    response = completion
                    break

        

        match = re.search(r'\$\\boxed{(.*?)}\$', response)
        n_samples += 1
        if match:
            model_answer = match.group(1)
            if true_answer == model_answer:
                n_true_ans += 1
                    
        
        table.append({"ID": i+1, 
                        "question": data["question"], 
                        "solution": response, 
                        "model_answer": model_answer,  
                        "answer": data['answer']})
        
        progress_bar.set_postfix(n_true_ans=n_true_ans)
    end = time.time()
    print("########################################################################################")
    print(f"Accuracy of LLaMA-3.2-1B-PRM@{N}: {n_true_ans/n_samples:.4f}.")
    print(f"Elapsed time for evaluating the base model is: {end-start:.2f} secs.")
    print("########################################################################################")

    if save_results:
        with open("./TTT_data/Best_of_" + str(N) + "_LLaMA-1B_Oracle.json", mode="w") as file:
            json.dump(table, file, indent=4)




#############################################################################################################
############################################## Majority Voting ##############################################
#############################################################################################################
def Majority_Voting(dataset, config: Config, llm: LLM, N=Config.n, save_results=False):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    index = 0
    
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1)
        x = {"question": [data['question']]}

        convs = [
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in x["question"]
        ]
        # TODO: set the augmented template from a file
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        # Duplicate convs to generate config.n completions per prompt so we can do continous batching
        # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
        templated_convs = [c for conv in templated_convs for c in [conv] * N]

        # Initialize empty lists for completions and completion tokens
        completions = [[] for _ in range(len(x["question"]))]
        completion_tokens = [[] for _ in range(len(x["question"]))]

        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(responses) != len(x["question"]) * N:
            raise ValueError(
                f"Generated {len(responses)} responses instead of {len(x['question'] * N)}"
            )

        for i in range(len(completions)):
            completions[i] = [
                output.text
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]
            completion_tokens[i] = [
                len(output.token_ids)
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]

        # Check we generated the correct number of completions for each prompt
        for c in completions:
            if len(c) != N:
                raise ValueError(f"Generated {len(c)} completions instead of {N}")

        model_responses = []
        response = completions[0][0]
        for completion in completions[0]:
            match = re.search(r'\\boxed\{(.*?)\}', completion)
            if match:
                model_responses.append(match.group(1))
            else:
                model_responses.append("")

        
        model_answer = ""
        element_counts = Counter(model_responses)
        dict_elements = dict(element_counts)
        sorted_elements = sorted(dict_elements, key=lambda x:dict_elements[x], reverse=True)
        
        if (len(model_responses) > 0):
            if (dict_elements[sorted_elements[0]] > N//2 and sorted_elements[0] != ""):
                indices = [i for i, val in enumerate(model_responses) if val == sorted_elements[0]]
                response = completions[0][indices[0]]

                response = response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
                match = re.search(r'\$\\boxed{(.*?)}\$', response)
                if match:
                    model_answer = match.group(1)
                    if true_answer == model_answer or true_answer.replace(" ", "") == model_answer.replace(" ", ""):
                        n_true_ans += 1


                table.append({"ID": index+1, 
                            "question": data["question"], 
                            "solution": response, 
                            "model_answer": model_answer,  
                            "answer": true_answer})
                index += 1
            else:
                response = response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
                match = re.search(r'\$\\boxed{(.*?)}\$', response)
                

                if match:
                    model_answer = match.group(1)
                    if true_answer == model_answer or true_answer.replace(" ", "") == model_answer.replace(" ", ""):
                        n_true_ans += 1
        
        n_samples += 1

            
        
        
        
        progress_bar.set_postfix(n_true_ans=n_true_ans)
    end = time.time()
    print("########################################################################################")
    print(f"Accuracy of LLaMA-3.2-1B-PRM@{N}: {n_true_ans/n_samples:.4f}.")
    print(f"Elapsed time for evaluating the base model is: {end-start:.2f} secs.")
    print("########################################################################################")

    if save_results:
        with open("./TTT_data/Best_of_" + str(N) + "Llama_1B_MV.json", mode="w") as file:
            json.dump(table, file, indent=4)




#############################################################################################################
############################################# Confidence based ##############################################
#############################################################################################################
def Confidence_Selection(dataset, config: Config, llm: LLM, N=Config.n, save_results=False):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    index = 0
    
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1)
        x = {"question": [data['question']]}

        convs = [
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in x["question"]
        ]
        # TODO: set the augmented template from a file
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        # Duplicate convs to generate config.n completions per prompt so we can do continous batching
        # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
        templated_convs = [c for conv in templated_convs for c in [conv] * N]

        # Initialize empty lists for completions and completion tokens
        completions = [[] for _ in range(len(x["question"]))]
        completion_tokens = [[] for _ in range(len(x["question"]))]

        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(responses) != len(x["question"]) * N:
            raise ValueError(
                f"Generated {len(responses)} responses instead of {len(x['question'] * N)}"
            )

        for i in range(len(completions)):
            completions[i] = [
                output.text
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]
            completion_tokens[i] = [
                len(output.token_ids)
                for r in responses[i * N : (i + 1) * N]
                for output in r.outputs
            ]

        # Check we generated the correct number of completions for each prompt
        for c in completions:
            if len(c) != N:
                raise ValueError(f"Generated {len(c)} completions instead of {N}")


        confidence_score = []
        for response in responses:
            completion = response.outputs[0]
            total_logprob = completion.cumulative_logprob
            confidence_score.append(total_logprob)

        max_log_likelihood = max(confidence_score)
        max_index = confidence_score.index(max_log_likelihood)
        response = completions[0][max_index]

        response = response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
        match = re.search(r'\$\\boxed{(.*?)}\$', response)
        n_samples += 1
        if match:
            model_answer = match.group(1)
            if true_answer == model_answer or true_answer.replace(" ", "") == model_answer.replace(" ", ""):
                n_true_ans += 1
        
        if max_log_likelihood > -100:
            table.append({"ID": index+1, 
                        "question": data["question"], 
                        "solution": response, 
                        "model_answer": model_answer,  
                        "answer": true_answer})
            index += 1
                
        
        
        
        
        progress_bar.set_postfix(n_true_ans=n_true_ans)
    end = time.time()
    print("########################################################################################")
    print(f"Accuracy of LLaMA-3.2-1B-PRM@{N}: {n_true_ans/n_samples:.4f}.")
    print(f"Elapsed time for evaluating the base model is: {end-start:.2f} secs.")
    print("########################################################################################")

    if save_results:
        with open("./TTT_data/Best_of_" + str(N) + "_Llama_1B_LogLikeLihood.json", mode="w") as file:
            json.dump(table, file, indent=4)




#############################################################################################################
################################################# Best Of N #################################################
#############################################################################################################
def best_of_n(x, config: Config, llm: LLM, prm: PRM, N=Config.n):
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["question"]
    ]
    tokenizer = llm.get_tokenizer()
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * N]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["question"]))]
    completion_tokens = [[] for _ in range(len(x["question"]))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    if len(responses) != len(x["question"]) * N:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['question'] * N)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * N : (i + 1) * N]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * N : (i + 1) * N]
            for output in r.outputs
        ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != N:
            raise ValueError(f"Generated {len(c)} completions instead of {N}")

    scores = prm.score(x["question"], completions)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    x["score"] = agg_scores[0][np.argmax(agg_scores)]

    return x

#############################################################################################################
################################ Evaluation: ["PRM" + "Pass@N" + "Verifier"] ################################
#############################################################################################################
def PRM_pass_N(x, config: Config, llm: LLM, prm: PRM, N, save_results=False):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""
        true_answer = ""
        input_batch = {"question": [data['question']]}
        response = best_of_n(input_batch, config, llm, prm, N)
        score = response["score"]
        response = response['pred'][0].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
        
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1)

        match = re.search(r'\$\\boxed{(.*?)}\$', response)
        n_samples += 1
        if match:
            model_answer = match.group(1)
            if true_answer == model_answer:
                n_true_ans += 1
       
        table.append({"ID": i+1, 
                        "question": data["question"], 
                        "solution": response, 
                        "score": score,
                        "model_answer": model_answer,  
                        "answer": true_answer})
        
        progress_bar.set_postfix(n_true_ans=n_true_ans)
    end = time.time()
    print("########################################################################################")
    print(f"Accuracy of LLaMA-3.2-1B-PRM@{N}: {n_true_ans/n_samples:.4f}.")
    print(f"Elapsed time for evaluating the base model is: {end-start:.2f} secs.")
    print("########################################################################################")

    if save_results:
        with open("./TTT_data/Best_of_" + str(N) + "_LLaMA_1B.json", mode="w") as file:
            json.dump(table, file, indent=4)
#############################################################################################################
########################################## Evaluation: ["Pass@1"] ###########################################
#############################################################################################################
def pass_1(dataset, config: Config, llm: LLM):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    n = 0
    n_true_ans = 0
    n_samples = 0
    model_answer = ""
    start = time.time()

    table = []
    progress_bar = tqdm(dataset, desc="Processing")
    for data in progress_bar:
        true_answer = data['answer']
        convs = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": data["question"]},
        ]
        # TODO: set the augmented template from a file
        
        templated_convs = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        response = responses[0].outputs[0].text
        response = response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()

        

        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1)

        match = re.search(r'\$\\boxed{(.*?)}\$', response)
        #match = re.search(r'\\boxed\{(.*?)\}', response)
        n_samples += 1
        if match:
            model_answer = match.group(1)
            if true_answer == model_answer or true_answer.replace(" ", "") == model_answer.replace(" ", ""):
                n_true_ans += 1

        table.append({"ID": n+1, 
                        "question": data["question"], 
                        "solution": response, 
                        "model_answer": model_answer,  
                        "answer": data['answer']})
        n += 1
        response = ""
        progress_bar.set_postfix(n_true_ans=n_true_ans)
    end = time.time()
    with open("./TTT_data/Inference_Generated_Results.json", mode="w") as file:
            json.dump(table, file, indent=4)
    print("########################################################################################")
    print(f"Base Model Accuracy is: {n_true_ans/n_samples:.4f}.")
    print(f"Elapsed time for evaluating the base model is: {end-start:.2f} secs.")
    print("########################################################################################")




#############################################################################################################
#############################################################################################################
#############################################################################################################
################################################### Main ####################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
if __name__ == "__main__":
    """
    model_path="meta-llama/Llama-3.2-1B-Instruct" | "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" | "deepseek-ai/deepseek-llm-7b-chat "
    model_path="./saved_models/merged_lora_model"
    model_path="./saved_models/Lora_with_Model/merged_lora_model_instruct_2/epoch_3/"             ## Lora + Instruct model Fine-Tuning
    model_path="./saved_models/Model_Without_Lora/4_Samples/epoch_1"    ## no Lora, Instruct model Fine-Tuning
    model_path="./saved_models/Lora_with_Model/MV_merged_lora_model_instruct_3/epoch_15/"    ## Moving Average With Lora
    model_path="./saved_models/Model_Without_Lora/MV_3_Samples/epoch_1"    ## Moving Average Without Lora
    prm_path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"                    ## Reward Model
    """
    config = Config()
    parser = argparse.ArgumentParser(description="Define hyperparameters for the model.")
    parser.add_argument("--n_repetitive_sampleing", default=4,                                              help="Number of samples the model generate")
    parser.add_argument("--temperature",            default=0.1,                                            help="temperature for setting the randomness (original github=0.8).")
    parser.add_argument("--top_p",                  default=1,                                              help="Top_p in the Generator")
    parser.add_argument("--model_path",             default="meta-llama/Llama-3.2-1B-Instruct",             help="base model")
    parser.add_argument("--prm_path",               default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",    help="Reward model path")
    parser.add_argument("--evaluation_choice",      default=2,                                              help="1-> original_model - 2-> Best_of_N_PRM - 3-> Oracle Verifier - 4-> Majority_Voting - 5-> Confidence_Selection")
    parser.add_argument("--save_to_json",           default=False,                                          help="Whethere to save the selected samples by verifier to the a JSON file")
    parser.add_argument("--dataset_repo_name",      default="openai/gsm8k",                                 help="Reward model path")
    args = parser.parse_args()
    
    config.n = args.n_repetitive_sampleing
    config.temperature = args.temperature
    config.top_p = args.top_p
    #config.system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $#### {answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

    ## <<<<< Loading the base model >>>>>
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.5,
        # tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        seed=42,
        dtype="float32",
        max_model_len=1600,
    )

    ## <<<<< Loading the Verifier >>>>>
    if (args.evaluation_choice == 2):
        prm = RLHFFlow(args.prm_path)

    ## <<<<< Loading the dataset >>>>>
    dataset = load_dataset(args.dataset_repo_name, 'main', split="test")

    if (args.evaluation_choice == 1):
        ## Original Model performance
        pass_1(dataset, config=config, llm=llm)

    elif (args.evaluation_choice == 2):
        ## Best_Of@N performance
        PRM_pass_N(dataset, config=config, llm=llm, prm=prm, N=config.n, save_results=args.save_to_json)
    
    elif (args.evaluation_choice == 3):
        ## Best_Of@N performance
        best_of_n_oracle(dataset, config=config, llm=llm, N=config.n, save_results=args.save_to_json)

    elif (args.evaluation_choice == 4):
        Majority_Voting(dataset, config=config, llm=llm, N=config.n, save_results=args.save_to_json)

    elif (args.evaluation_choice == 5):
        Confidence_Selection(dataset, config=config, llm=llm, N=config.n, save_results=args.save_to_json)

    for arg, value in vars(args).items():
        print(f"{arg} ===> {value}")


