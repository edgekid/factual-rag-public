from re import T
import random
import os
os.environ['HF_TOKEN']= "..."

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, get_linear_schedule_with_warmup, default_data_collator
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptEmbedding, PromptTuningConfig, TaskType, PeftType
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
import time
import matplotlib.pyplot as plt
#from accelerate import PartialState, prepare_pippy

from fuzzy_string_match import MATCH
from utils import exact_string_match, load_nq_long, load_narrativeqa, load_narrativeqa_test
# class MLP(nn.Module):

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_fn):
#         super(MLP, self).__init__()
#         self.mlp_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
#         for i in range(num_layers - 2):
#             self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
#         self.mlp_layers.append(nn.Linear(hidden_dim, output_dim))

#         self.act_fn = act_fn

#     def forward(self, x):
#         for i, l in enumerate(self.mlp_layers):
#             x = l(x)
#             if i != len(self.mlp_layers) - 1:
#                 x = self.act_fn(x)
#         return x


def get_gpu_with_most_free_memory():
    if not torch.cuda.is_available():
        return None, 0

    free_memory = []
    for i in range(torch.cuda.device_count()):
        free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory.append(free_mem)

    gpu_index = free_memory.index(max(free_memory))
    return gpu_index, free_memory[gpu_index]

class CaLCS(nn.Module):
    def __init__(self, temp = 1):
        super().__init__()
        self.softmax = nn.Softmax(dim = 2)
        self.alpha = temp

    def forward(self, batch, docs):
        # batch = batch_size x num_decoding_steps x vocabulary_size

        # Get p probabilities
        gpu_index, _ = get_gpu_with_most_free_memory()
        device = 'cuda:{}'.format(gpu_index)

        batch = self.softmax(batch / self.alpha).to(device)

        # Pad the documents so we can do batching for CaLCS computation
        batch = [b[:, docs[i]] for i, b in enumerate(batch)]
        mxlen = max([b.shape[-1] for b in batch])
        for i in range(len(batch)):
            batch[i] = torch.cat((torch.zeros(batch[i].shape[0], mxlen - batch[i].shape[-1]).to(device), batch[i]), dim=-1)
        batch = torch.stack(batch).to(device)
        # Compute s matrix
#        print(batch.shape)
#        s = torch.zeros(batch.shape[0], batch.shape[1] + 1, batch.shape[2] + 1).to(device)
#        m = s.shape[1]
#        k = s.shape[2]
#        v1 = torch.ones(s.shape[0]).to(device)
#        for j in range(1, m):
#            for i in range(1, k):
#                mx = torch.tensor(list(map(max, s[:, j - 1, i], s[:, j, i - 1]))).to(device)
#                s[:, j, i] = batch[:, j - 1, i - 1] * (s[:, j - 1, i - 1] + v1)
#                s[:, j, i] +=  (v1 - batch[:, j - 1, i - 1]) * mx
#
#        return -torch.log(s[:, m - 1, k - 1].mean() / mxlen)
#
        batch = torch.flip(batch, [1])
        batch_size = batch.size(0)
        s0 = torch.tensor([[]] * batch_size).to(device)
        s1 = torch.tensor([[]] * batch_size).to(device)

        r = batch.size(1)
        c = batch.size(2)
        for i in range(-r + 1, c):
            p = torch.stack([torch.diag(x, i) for x in batch]).to(device)
#            print('{}:'.format(p))
            if r < c:
                lens0 = s0.size(1)
                if i == -r + 1:
                    mx = torch.tensor([[0]] * batch_size)
                elif i <= 0:
                    mx = torch.cat((s0[:, :1], torch.max(s0[:, : -1], s0[:, 1 :]), s0[:, -1:]), dim=1)
                elif i <= -r + c:
                    mx = torch.cat((torch.max(s0[:, : -1], s0[:, 1 :]), s0[:, -1:]), dim=1)
                else:
                    mx = torch.max(s0[:, : -1], s0[:, 1 :])
                mx = mx.to(device)
                sc = torch.mul(torch.ones(p.size()).to(device) - p, mx)

                lens1 = s1.size(1)
                if i == -r + 1:
                    pc = torch.tensor([[x] for x in p[:, 0]])
                elif i <= 0:
                    pc = torch.cat((p[:, :1], torch.mul(s1 + torch.ones(batch_size, lens1).to(device), p[:, 1: -1]), p[:, -1:]), dim=1)
                elif i == 1:
                    pc = torch.cat((torch.mul(s1 + torch.ones(batch_size, lens1).to(device), p[:, : -1]), p[:, -1:]), dim=1)
                elif i <= -r + c:
                    pc = torch.cat((torch.mul(s1[:, 1:] + torch.ones(batch_size, lens1 - 1).to(device), p[:, : -1]), p[:, -1:]), dim=1)
                elif i == -r + c + 1:
                    pc = torch.mul(s1[:, 1: ] + torch.ones(batch_size, lens1 - 1).to(device), p)
                else:
                    pc = torch.mul(s1[:, 1: -1] + torch.ones(batch_size, lens1 - 2).to(device), p)

            else:
                lens0 = s0.size(1)
                if i == -r + 1:
                    mx = torch.tensor([[0]] * batch_size)
                elif i <= -r + c:
                    mx = torch.cat((s0[:, :1], torch.max(s0[:, : -1], s0[:, 1 :]), s0[:, -1:]), dim=1)
                elif i <= 0:
                    mx = torch.cat((s0[:, :1], torch.max(s0[:, : -1], s0[:, 1 :])), dim=1)
                else:
                    mx = torch.max(s0[:, : -1], s0[:, 1 :])
                mx = mx.to(device)
                sc = torch.mul(torch.ones(p.size()).to(device) - p, mx)

                lens1 = s1.size(1)
                if i == -r + 1:
                    pc = torch.tensor([[x] for x in p[:, 0]])
                elif i <= -r + c:
                    pc = torch.cat((p[:, :1], torch.mul(s1 + torch.ones(batch_size, lens1).to(device), p[:, 1: -1]), p[:, -1:]), dim=1)
                elif i == -r + c + 1:
                    pc = torch.cat((p[:, :1], torch.mul(s1 + torch.ones(batch_size, lens1).to(device), p[:, 1:])), dim=1)
                elif i <= 0:
                    pc = torch.cat((p[:, :1], torch.mul(s1[:, : -1] + torch.ones(batch_size, lens1 - 1).to(device), p[:, 1:])), dim=1)
                elif i == 1:
                    pc = torch.mul(s1[:, : -1] + torch.ones(batch_size, lens1 - 1).to(device), p)
                else:
                    pc = torch.mul(s1[:, 1: -1] + torch.ones(batch_size, lens1 - 2).to(device), p)

            pc = pc.to(device)
            sc = sc.to(device)
            s1 = s0
            s0 = pc + sc

        s0 = torch.min(s0, torch.tensor([100.] * s0.shape[0]).to(device)).to(device) # clipping the loss
        return -torch.log(s0.mean() / mxlen)

#class SoftPromptModel(nn.Module):
#    def __init__(self, num_tokens, model, tokenizer):
#        super().__init__()
#
#        self.tokenizer = tokenizer
#        self.model = model
#
#        self.emb_layer = self.model.get_input_embeddings()
#        emb_dim = self.emb_layer.embedding_dim
#
#        self.learnable_vectors = nn.Parameter(torch.randn(num_tokens, emb_dim))
#
#        # Freeze the model
#        for param in self.model.parameters():
#            param.requires_grad = False
#
#        # Projecting embedding coordinates of the llm_output's tokens to a's size
#        # self.logits_mlp = MLP(emb_dim, 128, len_a, 2, nn.ReLU())
#
#        self.n = num_tokens
#
#    def forward(self, batch):
#        default_token = self.tokenizer.pad_token_id
#        logits = []
#
#        for input in tqdm(batch):
#           # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
#            messages = [
#                {"role": "user", "content": input}
#            ]
#
#            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#            token_ids = self.tokenizer(input_ids, add_special_tokens=False, return_tensors="pt").to("cuda")
#
#            # Create a tensor of default tokens
#            default_tokens = torch.full((token_ids['input_ids'].size(0), self.n), default_token, dtype=torch.long).to("cuda")
#
#            # Append default tokens to the input IDs
#            token_ids['input_ids'] = torch.cat([default_tokens, token_ids['input_ids']], dim=1)
#            token_ids['attention_mask'] = torch.cat([torch.zeros_like(default_tokens), token_ids['attention_mask']], dim=1)
#
#            token_embeddings = self.emb_layer(token_ids['input_ids'])
#            token_embeddings[:self.n] = token_embeddings[:self.n] + self.learnable_vectors
#            # start = time.time()
#            outputs = self.model(inputs_embeds = token_embeddings, )
#            # end = time.time()
#            logits.append(outputs.logits)
#
#        return logits
#        # print('Output generated in {} sec'.format(end - start))


def pad_inputs(batch, length):
    for i in range(len(batch["input_ids"])):
        sample_input_ids = batch["input_ids"][i]
        batch["input_ids"][i] = torch.cat([sample_input_ids, torch.tensor([tokenizer.pad_token_id] * (length - len(sample_input_ids)), dtype=torch.int32).to("cuda")], dim = 0)
        batch["attention_mask"][i] = torch.cat([batch["attention_mask"][i], torch.tensor([0] * (length - len(sample_input_ids)), dtype=torch.int32).to("cuda")], dim = 0)
    batch['input_ids'] = torch.stack(batch['input_ids']).to("cuda")
    batch['attention_mask'] = torch.stack(batch['attention_mask']).to("cuda")
    return batch

def tensor_remove_rows(mat, rows):
    return torch.stack([r for num, r in enumerate(mat) if num not in rows]) if len(rows) != mat.shape[0] else torch.tensor([])

def generate_token_by_token(model, batch, max_new_tokens = 200, debug = False):
    model.eval()
    max_tokens = max_new_tokens
    length = batch['input_ids'].shape[1]
    prompt_length = length + params['num_tokens']
    batch_size = len(batch['input_ids'])
    texts = {'input_ids': [], 'attention_mask': []}
    pkv = None

    emb_layer = model.get_input_embeddings()
    base_model = model.base_model

    for tok in range(max_tokens):

        # Get the model's predictions for the next token (gradients will not be tracked)
        with torch.no_grad():
            # Get the model's predictions for the next token (gradients will not be tracked)
            if pkv == None:
                 token_embeddings = torch.cat([model.get_prompt(batch_size), emb_layer(batch['input_ids'])], dim = 1).to(torch.float16)
                 outputs = base_model(inputs_embeds = token_embeddings, attention_mask = torch.cat([torch.ones(batch_size, params['num_tokens']).to(torch.int32).to("cuda"), batch['attention_mask']], dim=-1), use_cache = True)
            else:
                 outputs = base_model(next_token_id, attention_mask = torch.cat([torch.ones(batch_size, params['num_tokens']).to(torch.int32).to("cuda"), batch['attention_mask']], dim=-1), use_cache = True, past_key_values = pkv)

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
    return pad_inputs(texts, length)

def train(model, train_loader, eval_loader, metrics_loader, params, loss_fn, documents, debug = True):
    # model.reset_parameters()
#    accelerator = Accelerator()

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * params["num_epochs"]),
    )

#    model = model.to("cuda")
#    train_loader, eval_loader, model, optimizer = accelerator.prepare(
#        train_loader, eval_loader, model, optimizer
#    )

    losses = []
    eval_losses = []
    exact_scores = []
    fuzzy_scores = []
    for epoch in tqdm(range(params["num_epochs"])):
        total_loss = 0

        st_0 = time.time()
        for step, batch in enumerate(train_loader):
            docs = [tokenizer.encode(documents[idx], add_special_tokens=False, return_tensors='pt').squeeze() for idx in batch['doc']]
            batch_w_targets = {k[2:]: v.to("cuda") for k, v in batch.items() if k != 'doc' and k[:2] == 't_'}
            batch = {k: v.to("cuda") for k, v in batch.items() if k != 'doc' and k[:2] != 't_'}
            prompt_length = batch['input_ids'].shape[1]

            # Compute prediction and loss
            st = time.time()
#            print(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True))
#            print(tokenizer.batch_decode(docs, skip_special_tokens = True))
#            print(torch.cuda.memory_allocated("cuda:0"))
            # Generate the whole sequence with prepended virtual tokens
            batch = generate_token_by_token(model, batch)
#            print(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True))
#            print(tokenizer.batch_decode(batch["input_ids"][:, prompt_length:], skip_special_tokens = True))
#            print(torch.cuda.memory_allocated("cuda:0"))

            # Find logits of the generated text and apply loss function
            model.train()
            optimizer.zero_grad()
            outputs = model(**batch)
#            print(torch.cuda.memory_allocated("cuda:0"))
            if params["lambda"] < 1:
                outputs_w_targets = model(**batch_w_targets)
#            print(torch.cuda.memory_allocated("cuda:0"))

            end = time.time()
            print('Prediction computed in {} sec'.format(end - st))

            st = time.time()
            lambd = params["lambda"]
            # CaLCS loss
            calcs = loss_fn(outputs.logits[:, (prompt_length + params["num_tokens"]):, :], docs).to("cuda")
            loss = lambd * calcs

            # language modeling loss
            lm_loss = torch.tensor(0.)
            if params["lambda"] < 1: 
                lm_loss = outputs_w_targets.loss
                loss += (1 - lambd) * lm_loss

            del outputs
            end = time.time()
            print('Loss {} (comrpised of {} + {}) computed in {} sec'.format(loss.detach().float().to("cpu"), calcs.detach().float().to("cpu"), lm_loss.detach().float().to("cpu"), end - st))

            del batch

            # Backpropagation
            st = time.time()
            loss.backward()
#            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            end = time.time()
            print('Backpropagation computed in {} sec'.format(end - st))

            total_loss += loss.detach().float().to("cpu")

        end_0 = time.time()
        print('Training phase completed in {}'.format(end_0 - st_0))
        # Evaluate metrics
        if epoch % 1 == 0:
            st_0 = time.time()
            eval_score = 0
            exact_score = 0
            nq = len(metrics_loader)
            for step, batch in enumerate(metrics_loader):
                docs = [documents[idx] for idx in batch['doc']]
                batch = {k: v.to("cuda") for k, v in batch.items() if k != "doc" and k[:2] != 't_'}
                batch_size = len(batch['input_ids'])
                prompt_length = batch['input_ids'].shape[1]

                st = time.time()
                outputs = generate_token_by_token(model, batch)
                temp_outputs = outputs['input_ids']
#                print(tokenizer.batch_decode(outputs['input_ids'], skip_special_tokens = True))
                outputs = tokenizer.batch_decode(outputs['input_ids'][:, prompt_length:], skip_special_tokens = True)
                batch_score = 0
                batch_score_0 = 0
                num_texts = 0

                for idx, out in enumerate(outputs):
                    print(tokenizer.decode(temp_outputs[idx], skip_special_tokens = True))
                    print("============================")
                    doc = docs[idx]
                    print(doc)
                    fuzzy, _ = MATCH(out.split(" "), doc.split(" "), tau = 4)
                    batch_score += fuzzy
                    exact = exact_string_match(out, doc)
                    batch_score_0 += exact
                    print('Fuzzy string match score: {}'.format(fuzzy))
                    print('Exact string match score: {}'.format(exact))

                    num_texts += 1

                eval_score += batch_score / num_texts
                exact_score += batch_score_0 / num_texts

                end = time.time()
                print('Score computed in {} sec'.format(end - st))
            end_0 = time.time()
            print('Score evaluation phase completed in {}'.format(end_0 - st_0))

        # Compute validation accuracy
        model.eval()
        st_0 = time.time()
        eval_loss = 0
        for step, batch in enumerate(eval_loader):
            docs = [tokenizer.encode(documents[idx], add_special_tokens=False, return_tensors='pt').squeeze() for idx in batch['doc']]
            batch_w_targets = {k[2:]: v.to("cuda") for k, v in batch.items() if k != 'doc' and k[:2] == 't_'}
            batch = {k: v.to("cuda") for k, v in batch.items() if k != 'doc' and k[:2] != 't_'}
            prompt_length = batch['input_ids'].shape[1]

            # Compute prediction and loss
            st = time.time()
            # Generate the whole sequence with prepended virtual tokens
            batch = generate_token_by_token(model, batch)
            # Find logits of the generated text and apply loss function
            with torch.no_grad():
                outputs = model(**batch)
                if params["lambda"] < 1:
                    outputs_w_targets = model(**batch_w_targets)

            end = time.time()
            print('Prediction computed in {} sec'.format(end - st))

            st = time.time()

            lambd = params["lambda"]
            # CaLCS loss
            calcs = loss_fn(outputs.logits[:, (prompt_length + params["num_tokens"]):, :], docs).to("cuda")
            loss = lambd * calcs

            # language modeling loss
            if params["lambda"] < 1:
                lm_loss = outputs_w_targets.loss
                loss += (1 - lambd) * lm_loss

            del outputs
            end = time.time()
            print('Loss {} computed in {} sec'.format(loss.detach().float().to("cpu"), end - st))

            del batch

            eval_loss += loss.detach().float().to("cpu")

        total_loss /= len(train_loader)
        eval_loss /= len(eval_loader)
        losses.append(total_loss)
        eval_losses.append(eval_loss)
        fuzzy_scores.append(eval_score / nq)
        exact_scores.append(exact_score / nq)
        end_0 = time.time()
        print('Testing phase completed in {}'.format(end_0 - st_0))

        print('Epoch {}: Training Loss {}, Eval Loss {}, Fuzzy Score {}, Exact Score {}'.format(epoch, total_loss, eval_loss, eval_score / nq, exact_score / nq))

    return model, losses, eval_losses, fuzzy_scores, exact_scores

def train_tokenwise(model, train_loader, eval_loader, params, loss_fn, doc, debug = True):
    # model.reset_parameters()
#    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * params["num_epochs"]),
    )

    q = next(iter(train_loader))
    q = {k: q[k][0].unsqueeze(dim = 0).to("cuda") for k in q.keys()}

#    model = model.to("cuda")
#    train_loader, eval_loader, model, optimizer = accelerator.prepare(
#        train_loader, eval_loader, model, optimizer
#    )

    losses = []
#    print(train_loader[0])
    for epoch in tqdm(range(params["num_epochs"])):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # Generate text token by token using greedy decoding, keeping gradients
#            optimizer.zero_grad()
            total_loss = 0

            max_tokens = 200
            init_len = length = batch['input_ids'].shape[1]
            prompt_length = length + params['num_tokens']
            batch_size = len(batch['input_ids'])
            pkv = None

            st = time.time()

            for tok in range(max_tokens):
                # Get the model's predictions for the next token
                optimizer.zero_grad()
                outputs = model(**batch)

                # Get the logits for the last token and find the token with the highest probability (greedy decoding)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to("cuda")

#                pkv = outputs.past_key_values

                # Append the next token to the output sequence
                batch['input_ids'] = torch.cat([batch['input_ids'], next_token_id], dim=-1).to("cuda")
                batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones(batch_size, 1).to(torch.int32).to("cuda")], dim=-1).to("cuda")

                if length != init_len:
                    loss = loss_fn(outputs.logits[:, prompt_length:, :], "cuda").to("cuda")
                    total_loss += loss.detach().float().to("cpu")
                    loss.backward()
                    optimizer.step()

                del outputs

                # Stop if the model generates the end-of-sequence token
                ind = []
                for i, tkn in enumerate(next_token_id):
                    if tkn in tokenizer.all_special_ids:
                        ind = ind + [i]

                batch['input_ids'] = tensor_remove_rows(batch['input_ids'], ind)
                batch['attention_mask'] = tensor_remove_rows(batch['attention_mask'], ind)

                next_token_id = tensor_remove_rows(next_token_id, ind).to("cuda")
                batch_size -= len(ind)
                # Delete the ith training data from the past_key_values matrices
#                if len(ind) > 0:
#                    tmp = ()
#                    for t in pkv:
#                        tmp = tmp + ((tensor_remove_rows(t[0], ind), tensor_remove_rows(t[1], ind)),)
#                    pkv = tmp

                length += 1

                if batch_size == 0:
                    break

            total_loss /= (length - init_len)
            end = time.time()
            print('Backpropagation computed in {} sec'.format(end - st))

#            optimizer.step()
            lr_scheduler.step()

        # Compute train accuracy
        model.eval()
        eval_loss = 0
        eval_preds = []

        st = time.time()
        outputs = generate_token_by_token(model, q.copy())
        outputs = tokenizer.batch_decode(outputs['input_ids'][:, init_len:], skip_special_tokens = True)
        print(outputs[0])
        print('Fuzzy string match score: {}'.format(MATCH(outputs[0], doc, tau = 12)))
        print('Exact string match score: {}'.format(exact_string_match(outputs[0], doc)))

        end = time.time()
        print('Score computed in {} sec'.format(end - st))

#        for step, batch in enumerate(eval_loader):
#            batch = {k: v.to("cuda") for k, v in batch.items()}
#
#            with torch.no_grad():
#                pred = model(**batch).logits
#            loss = loss_fn(pred)
#            eval_loss += loss.detach().float()
#            eval_preds.append(loss.detach().float())

        print('Epoch {}: Training Loss {}, Eval Loss {}'.format(epoch, total_loss / len(train_loader), eval_loss))

if __name__=='__main__':
    print(torch.__version__, torch.cuda.device_count())

    # Define the hyperparameters we are gonna use:
    params = {
        "learning_rate": 1e-3,
        "num_epochs": 10,
        "batch": 6,
        "num_tokens": 64,
        "temperature": 1,
        "lambda": 0.5
    }
    num_docs = 10
    no_targets = False

    models = ["google/gemma-7b-it", "meta-llama/Llama-2-7b-chat-hf"]
    model_name = models[1]
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM, #PromptTuningInit.TEXT,
        num_virtual_tokens=params["num_tokens"],
#        prompt_tuning_init_text="Answer the question with information extracted verbatim from the document.",
        tokenizer_name_or_path=model_name
    )

    # Load the tokenizer and model (replace with the actual model identifier)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

#    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for param in model.parameters():
        param.requires_grad = False

#    print(model.print_trainable_parameters())
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    print(model.hf_device_map)

    train_dataset = []
    test_dataset = []
    documents = []
    ans = []
    ds_metrics = []

    if no_targets:
        for i in range(num_docs):
            with open('data/data_unseen/d{}.txt'.format(i), 'r',  encoding="utf-8") as file:
                d = file.read()
                documents.append(d)
        #    with open('data/data_unseen/q{}.txt'.format(0), 'r',  encoding="utf-8") as file:
        #        qs = file.readlines()[7:8]
            with open('data/prompt_optimization/q{}.txt'.format(i), 'r',  encoding="utf-8") as file:
                qs = file.readlines()
            dataset += [(q, i, "") for q in qs]
    else:
        num_samples = 1000
        num_train = num_samples * 9 // 10
        num_test = num_samples - num_train
#        ds = load_nq_long(num_samples)
        ds = load_narrativeqa(num_train)
        for i in range(num_train):
            documents.append(ds["documents"][i])
            train_dataset += [(ds["questions"][i], i, ds["answers"][i])]

        ds = load_narrativeqa_test(num_test)
        for i in range(num_test):
            documents.append(ds["documents"][i])
            l = len(documents)
            test_dataset += [(ds["questions"][i], l - 1, ds["answers"][i])]

    # Getting documents to perform metric evaluation on
    for i in range(num_docs):
        with open('data/data_unseen/d{}.txt'.format(i), 'r',  encoding="utf-8") as file:
            d = file.read()
            documents.append(d)
            l = len(documents)
        with open('data/data_unseen/q{}.txt'.format(i), 'r',  encoding="utf-8") as file:
            qs = file.readlines()
        ds_metrics += [(q, l - 1, "") for q in qs]

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    def gen_train():
        for l in train_dataset:
            yield {"text": l[0], "d": l[1], "ans": l[2]}
    def gen_eval():
        for l in test_dataset:
            yield {"text": l[0], "d": l[1], "ans": l[2]}
    def gen_metrics():
        for l in ds_metrics:
            yield {"text": l[0], "d": l[1], "ans": l[2]}

    ds_train = Dataset.from_generator(gen_train)
    ds_eval = Dataset.from_generator(gen_eval)
    ds_metrics = Dataset.from_generator(gen_metrics)

    def preprocess_function(examples):
        messages = [[{"role": "user", "content": '{}In order to answer the question, you will be given a document of text. Output your answer without explaining why you chose this answer and without adding extra words.\nThis is the document:\n{}'.format(input, documents[examples["d"][i]])}] for i, input in enumerate(examples["text"])]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(input_ids, add_special_tokens=False)
        model_inputs["doc"] = examples["d"]
        max_length = max([len(input) for input in model_inputs['input_ids']])
#        max_length = 1660

        # Targets
        answers = [str(x) for x in examples["ans"]]
        model_targets = tokenizer(answers, add_special_tokens=False)
        max_length_t = max([len(input) for input in model_targets['input_ids']])
#        max_length_t = 50
        model_targets["labels"] = model_targets["input_ids"].copy()
        model_inputs["t_input_ids"] = model_targets["input_ids"]
        model_inputs["t_attention_mask"] = model_targets["attention_mask"]
        model_inputs["t_labels"] = model_targets["labels"]

        for i in range(len(examples["text"])):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]

            # Targets
            sample_target_ids = model_targets["input_ids"][i]
            model_targets["attention_mask"][i] = [0] * (max_length_t - len(sample_target_ids)) + model_inputs["attention_mask"][i] + [1] * len(sample_target_ids)
            model_targets["input_ids"][i] = [tokenizer.pad_token_id] * (max_length_t - len(sample_target_ids)) + model_inputs["input_ids"][i] + sample_target_ids
            model_targets["labels"][i] = [-100] * (len(model_inputs["input_ids"][i]) + max_length_t - len(sample_target_ids)) + sample_target_ids

            model_inputs["t_input_ids"][i] = torch.tensor(model_targets["input_ids"][i][:(max_length_t + max_length)])
            model_inputs["t_attention_mask"][i] = torch.tensor(model_targets["attention_mask"][i][:(max_length_t + max_length)])
            model_inputs["t_labels"][i] = torch.tensor(model_targets["labels"][i][:(max_length_t + max_length)])
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])

        return model_inputs

    train_dataset = ds_train.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=ds_train.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    eval_dataset = ds_eval.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=ds_eval.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    metrics_dataset = ds_metrics.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=ds_metrics.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    loss_fn = CaLCS(temp = params["temperature"])

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=params["batch"], pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=params["batch"], pin_memory=True)
#    metrics_dataloader = DataLoader(metrics_dataset, collate_fn=default_data_collator, batch_size=params["batch"], pin_memory=True)
    
    metrics_dataloader = eval_dataloader
 
    model, train_l, eval_l, fuzzy_s, exact_s = train(model, train_dataloader, eval_dataloader, metrics_dataloader, params, loss_fn, documents)
   
    # Plot train_l, eval_l, and exact_s
    plt.plot(train_l, label='Train loss')
    plt.plot(eval_l, label='Test loss')
    plt.plot(exact_s, label='Exact match score')
    plt.ylim(bottom=0)  # Set y-axis minimum to 0
    plt.legend()
    plt.savefig('plots/ans-1000-64-clip.png', bbox_inches = 'tight')
    plt.clf()

    plt.plot(fuzzy_s, label='Fuzzy match score')
    plt.ylim(bottom=0, top=max(fuzzy_s) * 1.2)
    plt.legend()
    plt.savefig('plots/ans-1000-64-clip-fuzzy.png', bbox_inches = 'tight')

    # save locally
    model.save_pretrained("models/joint_64_clip")
#    train_tokenwise(model, train_dataloader, eval_dataloader, params, loss_fn, d)

