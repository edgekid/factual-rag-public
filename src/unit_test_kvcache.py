import os
os.environ['HF_TOKEN']= "..."

import torch
from time import time
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, get_linear_schedule_with_warmup, default_data_collator
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptEmbedding, PromptTuningConfig, TaskType, PeftType

if __name__=='__main__':

    # Define the hyperparameters we are gonna use:
    ntokens = 30
    llama = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama2 = "meta-llama/Llama-2-7b-chat-hf"
    google = "google/gemma-7b-it"
    model_name = llama2

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=ntokens,
        prompt_tuning_init_text="Output should contain as much information found in the document as possible.",
        tokenizer_name_or_path=model_name,
    )

    # Load the tokenizer and model (replace with the actual model identifier)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
#        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
#        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = get_peft_model(model, peft_config)

    input = [[{"role": "user", "content": 'How many teeth do sharks have?'}]]
    input_ids = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
    init = model_inputs = tokenizer(input_ids, add_special_tokens=False, return_tensors="pt")['input_ids']

    max_tokens = 150
    pkv = None
   
    logits = []
    emb_layer = model.get_input_embeddings()
    base_model = model.base_model

    # With caching
    print('First 2 predictions with caching')
    st = time()
    for tok in range(max_tokens):

        token_embeddings = torch.cat([model.get_prompt(model_inputs.size(0)), emb_layer(model_inputs)], dim = 1).to(torch.float16)

        # Get the model's predictions for the next token (gradients will not be tracked)
        with torch.no_grad():
#             outputs = base_model(inputs_embeds = token_embeddings, attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1) + ntokens))
            if pkv == None:
                outputs = base_model(inputs_embeds = token_embeddings, attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1) + ntokens), use_cache = True)
            else:
                outputs = base_model(next_token_id, attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1) + ntokens), use_cache = True, past_key_values = pkv)

        # Get the logits for the last token and find the token with the highest probability (greedy decoding)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to("cpu")

        pkv = outputs.past_key_values

        # Append the next token to the output sequence
        model_inputs = torch.cat([model_inputs, next_token_id], dim=-1)
        del outputs

        # Stop if the model generates the end-of-sequence token
        ind = []
        for i, tkn in enumerate(next_token_id):
            if tkn in tokenizer.all_special_ids:
                break
    print('Generation done in {} sec'.format(time() - st))

    print(tokenizer.batch_decode(model_inputs, skip_special_tokens = True))
    logits.append(next_token_logits)

    print('First 2 predictions without caching')
    # Without caching
    model_inputs = init
    st = time()
    for tok in range(max_tokens):

        # Get the model's predictions for the next token (gradients will not be tracked)
        with torch.no_grad():
            outputs = model(model_inputs, attention_mask = torch.ones(model_inputs.shape))

        # Get the logits for the last token and find the token with the highest probability (greedy decoding)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to("cpu")

        # Append the next token to the output sequence
        model_inputs = torch.cat([model_inputs, next_token_id], dim=-1)

        del outputs

        # Stop if the model generates the end-of-sequence token
        ind = []
        for i, tkn in enumerate(next_token_id):
            if tkn in tokenizer.all_special_ids:
                break
    print('Generation done in {} sec'.format(time() - st))

    print(tokenizer.batch_decode(model_inputs, skip_special_tokens = True))
    logits.append(next_token_logits)

    print('First 2 predictions with num_logits_to_keep')
    # Without caching
    model_inputs = init
    st = time()
    for tok in range(max_tokens):

        # Get the model's predictions for the next token (gradients will not be tracked)
        with torch.no_grad():
            outputs = model(model_inputs, attention_mask = torch.ones(model_inputs.shape), num_logits_to_keep = 1)

        # Get the logits for the last token and find the token with the highest probability (greedy decoding)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to("cpu")

        # Append the next token to the output sequence
        model_inputs = torch.cat([model_inputs, next_token_id], dim=-1)

        del outputs

        # Stop if the model generates the end-of-sequence token
        ind = []
        for i, tkn in enumerate(next_token_id):
            if tkn in tokenizer.all_special_ids:
                break
    print('Generation done in {} sec'.format(time() - st))

    print(tokenizer.batch_decode(model_inputs, skip_special_tokens = True))
    logits.append(next_token_logits)


    print('Max abs difference between with and without caching: {}'.format(torch.max(logits[0] - logits[1]).item()))
    print('Max abs difference between with and without num_logits_to_keep: {}'.format(torch.max(logits[2] - logits[1]).item()))

