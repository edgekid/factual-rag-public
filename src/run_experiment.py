# THIS SCRIPT ASSUMES NO MORE THAN 1 THREAD RUNS THE EXPERIMENT ON THE SAME MODEL - OTHERWISE CONCURRENCY-RELATED PROBLEMS MAY APPEAR SUCH AS RACE CONDITION

import os
os.environ['HF_TOKEN']= "..."

from utils import evaluate_prompt_tinyllama, evaluate_prompt, load_synthetic, load_narrativeqa_test, load_narrativeqa, load_multiple_docs
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel, PeftConfig
import torch
import csv
from peft import get_peft_model
from answer_relevancy import CustomMistral7B

def project_onto_token_space(model, tokenizer, embeddings):
    # Get embedding matrix of model
    embedding_matrix = model.model.embed_tokens.weight

    embeddings = embeddings.unsqueeze(1)  # Shape: (num_vectors, 1, 4096)
    distances = torch.norm(embedding_matrix - embeddings, dim=2)

    # Find the indices of the minimum distance for each vector
    min_distance_indices = torch.argmin(distances, dim=1)
    closest_tokens = [tokenizer.decode([idx]) for idx in min_distance_indices]

    print("Closest tokens using Euclidean distance:", closest_tokens)

    return closest_tokens

def do_experiment(res_dir, models, datasets, models_mask, ds_mask, tokenizer, mistral_7b, compute_options):
    for i, tuple in enumerate(models):
        if not models_mask[i]:
            continue

        model = tuple[0]
        p = tuple[1]
        model_name = tuple[2]

        print("Running experiments on model {}".format(model_name))

        for j, data in enumerate(datasets):
            if not ds_mask[j]:
                continue

            print("Running experiments on dataset {}".format(data['name']))

            scores = evaluate_prompt(model, tokenizer, mistral_7b, data['documents'], data['questions'], p, compute_options['fuzzy'], compute_options['exact'], compute_options['ans_rel'], output_file = '{}/{}/{}/examples/'.format(res_dir, model_name, data['name']))

            for type in ['exact', 'fuzzy', 'ans_rel']:
                if not compute_options[type]:
                    continue

                file = '{}/{}/experiment_{}.csv'.format(res_dir, model_name, type)
                r = csv.reader(open(file)) # Read csv file
                lines = list(r)

                lines[1][j + 1] = scores[type].mean().item()

                with open(file, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerows(lines)

if __name__=='__main__':

    # Arguments

    # Which models to do the experiment on
    models_mask = [
        False, # base_llm
        False,
        False,
        False,
        False,
        False,
        False,
        False, # joint_64
        False, # calcs_only
        False, # token_proj_joint_64
        False, # joint_128
        False, # joint_64r
        True, # joint_64_clip
    ]

    datasets = [
        load_synthetic("seen"),
        load_synthetic("unseen"),
        load_synthetic("contradictory"),
        load_narrativeqa_test(50),
        load_multiple_docs(50)
    ]

    # Which datasets to do the experiment on
    ds_mask = [
        True,
        True,
        True,
        True,
        True
    ]

    models = ["google/gemma-7b-it", "meta-llama/Llama-2-7b-chat-hf"]

    mistral_7b = CustomMistral7B()

    res_dir = 'experiments'

    model_name = models[1]

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )


    if models_mask[7] or models_mask[9]:
        peft_model_id = "models/both_losses"
        config = PeftConfig.from_pretrained(peft_model_id)
        peft_joint_64 = {'model': PeftModel.from_pretrained(model, peft_model_id), 'virtual_tokens': config.num_virtual_tokens}
    else:
        peft_joint_64 = None

    if models_mask[8]:
        peft_model_id = "models/calcs_only"
        config = PeftConfig.from_pretrained(peft_model_id)
        peft_calcs = {'model': PeftModel.from_pretrained(model, peft_model_id), 'virtual_tokens': config.num_virtual_tokens}
    else:
        peft_calcs = None

    if models_mask[9]:
        embeddings = peft_joint_64['model'].get_prompt(1).squeeze(0)
        both_model_p = ""
        both_model_p = project_onto_token_space(model, tokenizer, embeddings)
        both_model_p = " ".join(both_model_p)
    else:
        both_model_p = None

    if models_mask[10]:
        peft_model_id = "models/joint_128"
        config = PeftConfig.from_pretrained(peft_model_id)
        peft_joint_128 = {'model': PeftModel.from_pretrained(model, peft_model_id), 'virtual_tokens': config.num_virtual_tokens}
    else:
        peft_joint_128 = None

    if models_mask[11]:
        peft_model_id = "models/joint_64-rand"
        config = PeftConfig.from_pretrained(peft_model_id)
        peft_joint_64r = {'model': PeftModel.from_pretrained(model, peft_model_id), 'virtual_tokens': config.num_virtual_tokens}
    else:
        peft_joint_64r = None

    if models_mask[12]:
        peft_model_id = "models/joint_64_clip"
        config = PeftConfig.from_pretrained(peft_model_id)
        peft_joint_64_clip = {'model': PeftModel.from_pretrained(model, peft_model_id), 'virtual_tokens': config.num_virtual_tokens}
    else:
        peft_joint_64r = None


    model = {'model': model, 'virtual_tokens': 0}
    models = [
        (model, "", "base_llm"),
        (model, "When answering the question, make sure to maximize the exact string match score, which is defined as repeatedly extracting the longest common substring between your answer and the document, squaring it and adding it to the final score.\n", "base_llm+p1"),
        (model, "When answering the question, copy paste as much relevant information from the document as you can.\n", "base_llm+p2"),
        (model, "Please include extensive relevant content from the document in your response.\n", "base_llm+p3"),
        (model, "Your response should include a substantial amount of relevant details from the document.\n", "base_llm+p4"),
        (model, "When answering the question, please make sure you have as many large common substrings between your answer and the document as you can.\n", "base_llm+p5"),
        (model, "Answer the question with information extracted verbatim from the document.", "base_llm+p6"),
        (peft_joint_64, "", "peft_joint_64"),
        (peft_calcs, "", "peft_calcs"),
        (model, both_model_p, "token_space_proj"),
        (peft_joint_128, "", "peft_joint_128"),
        (peft_joint_64r, "", "peft_joint_64r"),
        (peft_joint_64_clip, "", "peft_joint_64_clip")
    ]

    do_experiment(res_dir, models, datasets, models_mask, ds_mask, tokenizer, mistral_7b, compute_options = {'exact': True, 'fuzzy': True, 'ans_rel': True})
