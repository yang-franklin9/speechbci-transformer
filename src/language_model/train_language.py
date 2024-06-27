import re
import pickle
import os
import argparse

import torch
from .dataset_llm import getDatasetLoaders
from .llm_utils import llm, llm_embed, tokenize, tokenizer
from .config import device

from src.neural_decoder.neural_decoder_trainer_transformer import loadModel
from . import rnnEval

accum_steps = 16
num_batches = 10000 # number of effective batches to train for (1 * accum steps)
max_tokens = 50
hidden_size = 2048


torch.manual_seed(0)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--modelPath", type=str, default=None, help="Path to model")
input_args = parser.parse_args()


with open(input_args.modelPath + "/args", "rb") as handle:
    args = pickle.load(handle)

args["datasetPath"] = "data/ptDecoder_ctc"
trainLoader, testLoader, loadedData = getDatasetLoaders(
    args["datasetPath"], 1
)

model = loadModel(input_args.modelPath, device=device)
model.eval()

lin_proj = torch.nn.Sequential(torch.nn.Linear(512 * 5, hidden_size), torch.nn.LeakyReLU(), torch.nn.Linear(hidden_size, llm.config.hidden_size)).to(device)

# structure: embeds_1 proj_embs embeds_2 trscpt_embs
tokens_1 = tokenize("USER:")
if (tokenizer.bos_token_id != None):
    tokens_1 = [tokenizer.bos_token_id] + tokens_1
tokens_2 = tokenize("Transcribe speech to text. ASSISTANT:")
embeds_1 = llm_embed(torch.tensor(tokens_1, device = device))
embeds_2 = llm_embed(torch.tensor(tokens_2, device = device))

def calc_acc():
    with torch.no_grad():
        all_wer = []
        for X, y, X_len, y_len, testDayIdx in testLoader:
            X, y, X_len, y_len, testDayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                testDayIdx.to(device),
            )
            out = model.forward_no_cls(X, X_len)
            size = out.size(1) // 5
            out = torch.reshape(out[:, :size * 5], (out.size(0), size, -1))
            llm_in = lin_proj(out)
            llm_in = torch.cat((embeds_1.unsqueeze(0), llm_in, embeds_2.unsqueeze(0)), 1)
            llm_preds = []
            for id in range(max_tokens):
                llm_out = llm.forward(inputs_embeds = llm_in)
                # next token is from last logits of llm_out (i.e. prediction)
                next_tok = llm_out.logits[:, -1].argmax(-1)
                if (next_tok == tokenizer.eos_token_id):
                    break
                llm_preds.append(next_tok.item())
                llm_in = torch.cat((llm_in, llm_embed(next_tok).unsqueeze(0)), 1)

            targets = tokenizer.decode(y.squeeze(0), True, True)
            outputs = tokenizer.decode(torch.tensor(llm_preds), True, True)
            outputs = re.sub(r"[^a-zA-Z\s\']", "", outputs.lower())
            outputs = re.sub(r"\s+", " ", outputs)
            wer = rnnEval.wer(targets.split(), outputs.split()) / len(targets.split())
            all_wer.append(wer)

        return torch.tensor(all_wer).mean().item()


optimizer = torch.optim.Adam(lin_proj.parameters(), lr = 0.0005)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1, 100)

log = open("train_log.txt", "a")

breakpoint()

outputs = []
for i in range(num_batches):
    optimizer.zero_grad()
    batch_loss = 0
    for j in range(accum_steps):
        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
        )
        with torch.no_grad():
            # noise augmentation
            X += torch.randn(X.shape, device=device) * 0.8
            X += torch.randn([X.shape[0], 1, X.shape[2]], device=device) * 0.2

            out = model.forward_no_cls(X, X_len)
        
        # downsample
        size = out.size(1) // 5
        out = torch.reshape(out[:, :size * 5], (out.size(0), size, -1))

        llm_in = lin_proj(out)

        llm_in = torch.cat((embeds_1.unsqueeze(0), llm_in, embeds_2.unsqueeze(0)), 1)

        losses = []
        for k in range(y_len.item()):
            id = y[:, k]
            llm_out = llm.forward(inputs_embeds = llm_in)
            # compute cross entropy with last logits of llm_out (i.e. prediction)
            cur_loss = torch.nn.functional.cross_entropy(llm_out.logits[:, -1], id.long())
            next_emb = llm_embed(id)
            llm_in = torch.cat((llm_in, next_emb.unsqueeze(0)), 1)
            losses.append(cur_loss)
        loss = torch.stack(losses).mean()
        loss /= accum_steps
        loss.backward()
        batch_loss += loss

    optimizer.step()
    scheduler.step()

    if (i % 100 == 0):
        test_acc = calc_acc()
        print(i, batch_loss.item(), test_acc)

        print(i, batch_loss.item(), test_acc, file = log)
        log.flush()
        os.fsync(log.fileno())
        with open("proj", "wb") as f:
            pickle.dump(lin_proj, f)

with open("proj", "wb") as f:
    pickle.dump(lin_proj, f)
breakpoint()
