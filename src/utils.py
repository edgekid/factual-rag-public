import pylcs
import torch
from transformers import pipeline
import re
import time
import numpy as np
#from datasets import Dataset
import os
import random

from fuzzy_string_match import MATCH
from answer_relevancy import answer_relevancy

from datasets import load_dataset
import numpy as np

def load_nq_long(num_samples, max_length = 500):
    ds = load_dataset("google-research-datasets/natural_questions", "default")
    i = 0
    dataset = {'questions': [], 'documents': [], 'answers': []}
    while num_samples > 0:
        sample = ds['train'][i]
        i += 1
        if len(sample['annotations']['short_answers'][0]['start_token']) == 0:
            continue

        q = sample['question']['text'] + "?\n"

        is_html = np.array(sample['document']['tokens']['is_html'], dtype=bool)
        is_not_html = ~is_html
        doc_tokens = np.array(sample['document']['tokens']['token'])
        doc = " ".join(doc_tokens[is_not_html][:max_length])

        ans_token = sample['annotations']['long_answer'][0]['end_token']
        if len(doc_tokens[:ans_token][is_not_html[:ans_token]]) > max_length:
            continue

        ans_st = sample['annotations']['long_answer'][0]['start_token']
        ans = doc_tokens[ans_st: ans_token][is_not_html[ans_st: ans_token]]
        num_samples -= 1

        dataset['questions'].append(q)
        dataset['documents'].append(doc)
        dataset['answers'].append(ans)
    return dataset

def load_narrativeqa(num_samples, max_length = 10000):
    ds = load_dataset("deepmind/narrativeqa")
    i = 0
    dataset = {'name': 'NarrativeQA', 'questions': [], 'documents': [], 'answers': []}
    while num_samples > 0 and i < len(ds['train']):
        sample = ds['train'][i]
        i += 1

        q = sample['question']['text'] + "\n"

        doc = sample['document']['summary']['text']
        ans = ""
        for a in sample['answers']:
            if len(a['text']) > len(ans):
                ans = a['text']
        if len(ans) <= 50:
            continue

        num_samples -= 1

        dataset['questions'].append(q)
        dataset['documents'].append(doc)
        dataset['answers'].append(ans)

    return dataset

def load_synthetic(ds_name):

    valid = {"seen", "unseen", "contradictory"}
    if ds_name not in valid:
        raise ValueError("Error: ds_name must be one of %r." % valid)

    dataset = {'questions': [], 'documents': [], 'name': 'synthetic_{}'.format(ds_name)}
    for i in range(0, 10):
        with open('data/data_{}/d{}.txt'.format(ds_name, i), 'r',  encoding="utf-8") as file:
            doc = file.read()
        with open('data/data_{}/q{}.txt'.format(ds_name, i), 'r',  encoding="utf-8") as file:
            queries = file.readlines()

        dataset['questions'] += queries
        dataset['documents'] += [doc] * len(queries)
    return dataset

def load_narrativeqa_test(num_samples):
    lim_doc = 4000
    ds = load_dataset("deepmind/narrativeqa")
    i = 0
    dataset = {'questions': [], 'documents': [], 'answers': [], 'name': 'narrative_qa'}
    while num_samples > 0 and i < len(ds['test']):
        sample = ds['test'][i]
        i += 1

        q = sample['question']['text'] + "\n"

        doc = sample['document']['summary']['text']
        if len(doc) < lim_doc:
            continue

        ans = ""
        for a in sample['answers']:
            if len(a['text']) > len(ans):
                ans = a['text']
        if len(ans) <= 50:
            continue

        num_samples -= 1

        dataset['questions'].append(q)
        dataset['documents'].append(doc)
        dataset['answers'].append(ans)

    return dataset

# Creates a synthetic dataset with a question and 'num_docs' documents as context
# Some questions are related to only one of the document while others are not related to any at all
def load_multiple_docs(num_samples, num_docs = 3):
    ds = load_dataset("deepmind/narrativeqa")
    i = 1500
    max_doc = 2500
    min_doc = 1000
    n = num_samples * (num_docs + 1)
    qs = []
    docs = []
    while n > 0 and i < len(ds['train']):
        sample = ds['train'][i]
        i += 1

        q = sample['question']['text'] + "\n"

        doc = sample['document']['summary']['text']
        if len(doc) > max_doc or len(doc) < min_doc or doc in docs:
            continue
        
        n -= 1

        qs.append(q)
        docs.append(doc)

#    print(n, len(ds['train']))

    dataset = {'questions': [], 'documents': [], 'name': 'multiple_docs'}
    i = 0
    while i < num_samples:
        mdocs = ""
        queries = []

        for j in range(0, num_docs):
            mdocs += docs[i + j] + "\n"

        j = 0
        while i < num_samples and j <= num_docs:
            dataset['questions'].append(qs[i])
            dataset['documents'].append(mdocs)

            i += 1
            j += 1

    return dataset

def pad_inputs(batch, length, tokenizer):
    for i in range(len(batch["input_ids"])):
        sample_input_ids = batch["input_ids"][i]
        batch["input_ids"][i] = torch.cat([sample_input_ids, torch.tensor([tokenizer.pad_token_id] * (length - len(sample_input_ids)), dtype=torch.int32).to("cuda")], dim = 0)
        batch["attention_mask"][i] = torch.cat([batch["attention_mask"][i], torch.tensor([0] * (length - len(sample_input_ids)), dtype=torch.int32).to("cuda")], dim = 0)
    batch['input_ids'] = torch.stack(batch['input_ids']).to("cuda")
    batch['attention_mask'] = torch.stack(batch['attention_mask']).to("cuda")
    return batch

def tensor_remove_rows(mat, rows):
    return torch.stack([r for num, r in enumerate(mat) if num not in rows]) if len(rows) != mat.shape[0] else torch.tensor([])

def generate_token_by_token(model, tokenizer, batch, max_new_tokens = 200, debug = False):
    virtual_tokens = model['virtual_tokens']
    model = model['model']
    model.eval()
    max_tokens = max_new_tokens
    length = batch['input_ids'].shape[1]
    prompt_length = length + virtual_tokens
    batch_size = len(batch['input_ids'])
    texts = {'input_ids': [], 'attention_mask': []}
    pkv = None

    emb_layer = model.get_input_embeddings()
    base_model = model.base_model

    for tok in range(max_tokens):
        # Create a pipeline stage from the model
        # Using `auto` is equivalent to letting `device_map="auto"` figure
        # out device mapping and will also split the model according to the
        # number of total GPUs available if it fits on one GPU
#        model = prepare_pippy(model, split_points="auto", example_kwargs=batch)
#        batch = {k: v.to("cuda") for k, v inw batch.items()}

        # Get the model's predictions for the next token (gradients will not be tracked)
        with torch.no_grad():
            # Get the model's predictions for the next token (gradients will not be tracked)
            if pkv == None:
                 token_embeddings = torch.cat([model.get_prompt(batch_size), emb_layer(batch['input_ids'])], dim = 1).to(torch.float16)
                 outputs = base_model(inputs_embeds = token_embeddings, attention_mask = torch.cat([torch.ones(batch_size, virtual_tokens).to(torch.int32).to("cuda"), batch['attention_mask']], dim=-1), use_cache = True)
            else:
                 outputs = base_model(next_token_id, attention_mask = torch.cat([torch.ones(batch_size, virtual_tokens).to(torch.int32).to("cuda"), batch['attention_mask']], dim=-1), use_cache = True, past_key_values = pkv)

#        if not PartialState().is_last_process:
#            continue

        # Get the logits for the last token and find the token with the highest probability (greedy decoding)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to("cuda")

        pkv = outputs.past_key_values
        if debug:
            print(pkv)

        # Append the next token to the output sequence
        batch['input_ids'] = torch.cat([batch['input_ids'], next_token_id], dim=-1).to("cuda")
        batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones(batch_size, 1).to(torch.int32).to("cuda")], dim=-1).to("cuda")

        # Stop if the model generates the end-of-sequence token
        ind = []
        for i, tkn in enumerate(next_token_id):
            if tkn in tokenizer.all_special_ids:
                texts['input_ids'].append(batch['input_ids'][i])
                texts['attention_mask'].append(batch['attention_mask'][i])
                ind = ind + [i]

        batch['input_ids'] = tensor_remove_rows(batch['input_ids'], ind)
        batch['attention_mask'] = tensor_remove_rows(batch['attention_mask'], ind)

        next_token_id = tensor_remove_rows(next_token_id, ind).to("cuda")
        batch_size -= len(ind)
        # Delete the ith training data from the past_key_values matrices
        if len(ind) > 0:
            tmp = ()
            for t in pkv:
                tmp = tmp + ((tensor_remove_rows(t[0], ind), tensor_remove_rows(t[1], ind)),)
            pkv = tmp

        length += 1
        del outputs

        if batch_size == 0:
            break

    for i in range(batch_size):
        texts['input_ids'].append(batch['input_ids'][i])
        texts['attention_mask'].append(batch['attention_mask'][i])
#    print(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
    return pad_inputs(texts, length, tokenizer)


# Measures how much of text a can be found in text b. Longer common substrings weigh more.
def exact_string_match(a, b):
    if len(a) == 0:
        return 0

    def substring_weight(length):
        return length ** 2

    l = substring_weight(len(a))
    score = 0
    while True:
        res = pylcs.lcs_string_idx(a, b)

        marked = False
        new = ''
        for i in range(len(a)):
            if res[i] == -1:
                new += a[i]
            elif marked == False:
                marked = True
                new += '*#'
        # new = ''.join(a[i] for i in range(len(a)) if res[i] == -1)
        lcs = len(a) - len(new) + 2

        if lcs < 12:
            break

        # Eliminate the substring from a
        a = new

        score += substring_weight(lcs)
    return  score / l

# Returns the length and starting position (in a) of the longest common substring between a and b
# def longest_common_substring(a, b):
#     dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
#     cur_mx = 0
#     start_a = 0
#     for i in range(1, len(a) + 1):
#         for j in range(1, len(b) + 1):
#             if a[i - 1] == b[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#             if dp[i][j] > cur_mx:
#                 cur_mx = dp[i][j]
#                 start_a = i - cur_mx

#     return cur_mx, start_a

def evaluate_prompt_tinyllama(d, queries, p):
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=0)

    mistral_7b = CustomMistral7B()

    scores = []

    # print("PROMPT:{}\n".format(whole_page))

    for k, q in enumerate(queries):

        print("QUERY {}".format(k))

        augment_option = 0
        # for augment_option in range(4):
        if augment_option == 0:
            prompt = "You will be given a corpus of text and a question based on it." + p + "This is the question:\n" + q + "\nThis is the corpus:\n" + d
        #elif augment_option == 1:
        #    prompt = "In order to answer any question, consider the following information:\n" + d + "\nQuestion: " + q
        #elif augment_option == 2:
        #    prompt = "In order to answer any question, you may consider the following information:\n" + d + "\nQuestion: " + q
        elif augment_option == 3:
            prompt = q + "\nIn order to answer the question, you will be given a corpus of text." + p + "This is the corpus:\n" + d

        messages = [
            {"role": "user", "content": prompt}
        ]

        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        final_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        start = time.time()
        outputs = pipe(final_prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
        end = time.time()

        output = outputs[0]["generated_text"]

        print(output)
        print('Output generated in {} sec'.format(end - start))

        start = time.time()
        score = exact_string_match(output, d)
        print("Score: {}".format(score))
        scores.append(score)
        end = time.time()
        print('Exact string match calculated in {} sec\n'.format(end - start))

        answer_relevancy(mistral_7b, q, output, d)

    # avg_score /= len(queries)
    return scores
    # print("AVERAGE SCORE FOR PROMPT {}".format(avg_score))

def evaluate_prompt(model, tokenizer, ans_rel_model, documents, queries, p, compute_fuzzy, compute_exact, compute_ans_rel, output_file = None):
    batch_size = 1

    scores = []
    fuzzy_scores = []
    acc_scores = []

    for i in range(len(queries) // batch_size + bool((len(queries)%batch_size) != 0)):

        messages = []
        for j, q in enumerate(queries[i * batch_size : (i + 1) * batch_size]):
            d = documents[j + i * batch_size]
            prompt = '{}{}In order to answer the question, you will be given a document of text. Output your answer without explaining why you chose this answer and without adding extra words.\nThis is the document:\n{}'.format(p, q, d)

            messages.append([
                {"role": "user", "content": prompt}
            ])

        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_ids, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        prompt_length = input_ids['input_ids'].shape[1]

        start = time.time()
        if model['virtual_tokens'] == 0:
            outputs = model['model'].generate(**input_ids, max_new_tokens = 300)
        else:
            outputs = generate_token_by_token(model, tokenizer, input_ids, max_new_tokens = 300)['input_ids']
        end = time.time()

        output = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens = True)
        print(output)
        print('Output generated in {} sec'.format(end - start))

        for j, out in enumerate(output):
            start = time.time()

            idx = i * batch_size + j
            d = documents[idx]

            if output_file != None:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open('{}/q{}.out'.format(output_file, idx), 'w') as file:
                    file.write("Prompt:\n{}\nOutput:\n{}\n".format(tokenizer.decode(outputs[j][:prompt_length], skip_special_tokens = True), out))

            if compute_exact:
                score = exact_string_match(out, d)
                scores.append(score)
                print("Exact Score: {}".format(score))
                if output_file != None:
                    with open('{}/q{}.out'.format(output_file, idx), 'a') as file:
                        file.write("Exact Score: {}\n".format(score))

            if compute_fuzzy:
                score, largest_match = MATCH(out.split(" "), d.split(" "), tau = 4)
                fuzzy_scores.append(score)
                print("Fuzzy Score: {}".format(score))
                if output_file != None:
                    with open('{}/q{}.out'.format(output_file, idx), 'a') as file:
                        file.write("""Fuzzy Score: {} with largest match: "{}"\n""".format(score, largest_match))

            end = time.time()
            print('String match score calculated in {} sec\n'.format(end - start))

            if compute_ans_rel:
                ans_rel = answer_relevancy(ans_rel_model, q, out, d)
                acc_scores.append(ans_rel)
                if output_file != None:
                    with open('{}/q{}.out'.format(output_file, idx), 'a') as file:
                        file.write('Answer relevancy score: {}\n'.format(ans_rel))

    return {"exact": torch.tensor(scores, dtype = torch.float16), "ans_rel": torch.tensor(acc_scores, dtype = torch.float16), "fuzzy": torch.tensor(fuzzy_scores, dtype = torch.float16)}

