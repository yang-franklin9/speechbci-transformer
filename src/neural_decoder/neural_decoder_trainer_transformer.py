# Code modified with permission from cffan/neural_seq_decoder

import os
import pickle
import time
import math

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset
from .augmentations import GaussianSmoothing


class Transformer(torch.nn.Module):
    def __init__(self, neural_dim, n_classes, gaussianSmoothWidth, kernelLen, strideLen, dropout, conv_size = 512, d_model = 384, device = None):
        super().__init__()
        self.d_model = d_model
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = d_model, nhead = 4, dim_feedforward = 2048, dropout = dropout, activation = "gelu", batch_first = True, device = device)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers = 6)

        self.dim_encoding = torch.pow(10000, -torch.arange(0, 1, 2 / d_model, device = device))
        self.pos_drop = torch.nn.Dropout(dropout)

        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, gaussianSmoothWidth, dim=1
        )
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.kernelLen = kernelLen
        self.strideLen = strideLen

        self.input_conv = torch.nn.Conv1d(neural_dim, conv_size, kernelLen, strideLen)
        self.input_encode = torch.nn.Linear(conv_size, d_model, device = device)
        self.cls = torch.nn.Linear(d_model, n_classes + 1, device = device)

    def adj_len(self, l):
        l = (l - self.kernelLen) // self.strideLen + 1
        return l

    def add_pos_encoding(self, x):
        with torch.no_grad():
            encoding = torch.arange(x.size(1), device = x.device).unsqueeze(1) * self.dim_encoding
            pos_encoding = torch.zeros_like(x)
            pos_encoding[:, :, 0::2] = torch.sin(encoding)
            pos_encoding[:, :, 1::2] = torch.cos(encoding)
        x += pos_encoding
        return self.pos_drop(x)

    def forward_no_cls(self, src, src_size):
        with torch.no_grad():
            src = torch.permute(src, (0, 2, 1))
            src = self.gaussianSmoother(src)
            src = torch.permute(src, (0, 2, 1))
            src = self.inputLayerNonlinearity(src)

        # conv operates on [B, C, N]
        src = self.input_conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = torch.nn.functional.gelu(src)
        # Linear should operate on [B, N, C]
        src_tokens = self.input_encode(src)

        src_tokens = self.add_pos_encoding(src_tokens)

        src_pad_mask = torch.arange(src.size(1), device = src.device).expand(src.size(0), -1)
        length = src_size.unsqueeze(1).expand(-1, src.size(1))
        length = self.adj_len(length)
        src_pad_mask = length <= src_pad_mask

        return self.encoder(src_tokens, src_key_padding_mask = src_pad_mask)

    def forward(self, src, src_size):
        res = self.forward_no_cls(src, src_size)
        res = self.cls(res)
        return res

def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = Transformer(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrMax"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: math.sqrt(args["warmup"]) * min(1/math.sqrt(max(0.01, step)), step * math.pow(args["warmup"], -1.5)))

    # --train--
    trainLoss = []
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X, X_len)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            model.adj_len(X_len),
            y_len,
        )
        loss = torch.sum(loss)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        trainLoss.append(loss.item())

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, X_len)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        model.adj_len(X_len),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = model.adj_len(X_len)
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()

                avgTrainLoss = sum(trainLoss) / len(trainLoss)
                trainLoss = []

                print(
                    f"batch {batch}, train loss: {avgTrainLoss:>7f}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = Transformer(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
