
""" 
Training script for models.

Training some models on CIFAR10 for fun.
"""

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import data_utils 
import networks

NETWORKS = {
    "vgg": networks.get_vgg16_model,
    "resnet18": networks.get_resnet18_model,
    "resnet50": networks.get_resnet50_model,
    "inception": networks.get_inception_v3_model
}


class ModelTrainer:

    def __init__(self, args):
        assert args["model"] in NETWORKS.keys(), f"model should be one of {list(NETWORKS.keys())}"
        if not args.get("model_ckpt", None):
            self.model, self.model_until_pooling, self.classifier = NETWORKS[args["model"]](True, None)
        else:
            self.model, self.model_until_pooling, self.classifier = NETWORKS[args["model"]](False, torch.load(args["model_ckpt"]))
        
        if args["model"] == "vgg":
            self.model.classifier[-1] = nn.Linear(4096, 10, bias=True)
        elif args["model"] == "resnet18":
            self.model.fc = nn.Linear(512, 10, bias=True)
        else:
            self.model.fc = nn.Linear(2048, 10, bias=True)
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.00001)

        self.network_transform = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.normal_transform = T.Compose([T.Resize(128), T.ToTensor()])
        self.train_loader, self.val_loader, _ = data_utils.get_cifar10_dataloader(
            "./data", args["batch_size"], self.network_transform, self.normal_transform)

        self.criterion = nn.NLLLoss()
        self.best_val_acc = 0

    def train_one_step(self, batch):
        img, trg = batch 
        img, trg = img.to(self.device), trg.to(self.device)
        out = self.model(img)
        loss = self.criterion(F.log_softmax(out, dim=-1), trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        acc = preds.eq(trg.view_as(preds)).sum().item() / preds.numel()
        return {"loss": loss.item(), "accuracy": acc}

    def validate_one_step(self, batch):
        img, trg = batch 
        img, trg = img.to(self.device), trg.to(self.device)
        with torch.no_grad():
            out = self.model(img)
        loss = self.criterion(F.log_softmax(out, dim=-1), trg)
        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        acc = preds.eq(trg.view_as(preds)).sum().item() / preds.numel()
        return {"loss": loss.item(), "accuracy": acc}

    def save_model(self):
        state = self.model.state_dict()
        torch.save(state, f"./outputs/trained_models/cifar10_{self.args['model']}.ckpt")    

    def train(self, epochs=20, eval_every=2):
        print()
        for epoch in range(epochs):
            train_loss, train_acc = [], []
            for batch in tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                train_metrics = self.train_one_step(batch)
                train_loss.append(train_metrics["loss"])
                train_acc.append(train_metrics["accuracy"])
            print("[TRAIN] Epoch {:2d}/{} - [loss] {:.4f} [accuracy] {:.4f}".format(
                epoch+1, epochs, np.mean(train_loss), np.mean(train_acc)))

            if (epoch+1) % eval_every == 0:
                val_loss, val_acc = [], []
                for batch in tqdm(self.val_loader, total=len(self.val_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                    val_metrics = self.validate_one_step(batch)
                    val_loss.append(val_metrics["loss"])
                    val_acc.append(val_metrics["accuracy"])
                print("[VALID] Epoch {:2d}/{} - [loss] {:.4f} [accuracy] {:.4f}".format(
                    epoch+1, epochs, np.mean(val_loss), np.mean(val_acc)))

                if np.mean(val_acc) > self.best_val_acc:
                    self.best_val_acc = np.mean(val_acc)
                    self.save_model()               