
""" 
Main class to generate heatmaps.

Filename is misleading, but it's my convention.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T 
import cv2
import networks 
import data_utils
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime as dt
from PIL import Image
from tqdm import tqdm
import class_maps


NETWORKS = {
    "vgg": networks.get_vgg16_model,
    "resnet18": networks.get_resnet18_model,
    "resnet50": networks.get_resnet50_model,
    "inception": networks.get_inception_v3_model
}


class AblationCAM:

    def __init__(self, args):
        assert args["model"] in NETWORKS.keys(), f"model should be one of {list(NETWORKS.keys())}"
        if not args.get("model_ckpt", None):
            self.model, self.model_until_pooling, self.classifier = NETWORKS[args["model"]](True, None)
        else:
            self.model, self.model_until_pooling, self.classifier = NETWORKS[args["model"]](False, torch.load(args["model_ckpt"]))

        self.args = args
        self.method, self.act_fn, self.clf_act_fn = args["method"], args["final_activation"], args["clf_activation"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_until_pooling = self.model.to(self.device), self.model_until_pooling.to(self.device)
        self.classifier = self.classifier.to(self.device)

        # Dataloader
        if "imagenet" in args["data_root"]:
            self.network_transform = T.Compose([
                T.Resize(128), T.CenterCrop(128), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.normal_transform = T.Compose([
                T.Resize(128), T.CenterCrop(128), T.ToTensor()])
            self.network_loader = data_utils.get_imagenet_dataloader(
                args["data_root"], args["batch_size"], self.network_transform, self.normal_transform)

        else:
            self.network_transform = T.Compose([
                T.Resize(128), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.normal_transform = T.Compose([
                T.Resize(128), T.ToTensor()])
            self.network_loader, _, self.normal_loader = data_utils.get_cifar10_dataloader(
                args["data_root"], args["batch_size"], self.network_transform, self.normal_transform)

    def generate_localization_map(self, batch, method="channel", final_activation="relu", class_activation="softmax"):
        if "imagenet" in self.args["data_root"]:
            img, _, trg = batch 
        elif "cifar" in self.args["data_root"]:
            img, trg = batch

        img, trg = img.to(self.device), trg.to(self.device)
        trg_cpu = trg.detach().clone().cpu().numpy()
        with torch.no_grad():
            logits, pooler_out = self.model(img), self.model_until_pooling(img)                         # (bs, 1000), (bs, c, h, w)
            probs = torch.sigmoid(logits)
        yc = torch.gather(probs, 1, trg.view(-1, 1)).contiguous().view(1, -1)                           # (1, bs)
        bs, c, h, w = pooler_out.size()

        # Predictions of the model
        preds = probs.argmax(dim=1).cpu().numpy()
        
        if "imagenet" in self.args["data_root"]:
            pred_names = [(class_maps.imagenet_class_map[idx], probs[i, idx].item()) for i, idx in enumerate(preds)]
            true_names = [class_maps.imagenet_class_map[idx] for idx in trg_cpu]
        elif "cifar" in self.args["data_root"]:
            pred_names = [(class_maps.cifar10_class_map[idx], probs[i, idx].item()) for i, idx in enumerate(preds)]
            true_names = [class_maps.cifar10_class_map[idx] for idx in trg_cpu]

        # Ablation, "channel" for channel-wise and "spatial" for location-wise 
        yck_collect = []

        if method == "channel":                                                         # Channel wise ablation as done in the paper
            for c_idx in range(pooler_out.size(1)):
                ablation_map = pooler_out.detach().clone()
                ablation_map[:, c_idx, :, :] = 0.
                with torch.no_grad():
                    if class_activation == "sigmoid":
                        changed_probs = torch.sigmoid(self.classifier(ablation_map))
                    elif class_activation == "softmax":
                        changed_probs = F.softmax(self.classifier(ablation_map), dim=-1)
                yck_collect.append(torch.gather(changed_probs, 1, trg.view(-1, 1)).contiguous().view(1, -1))
            
            # Compute weights and activated localization map
            yck_collect = torch.cat(yck_collect, dim=0)                                             # (c, bs) 
            yc = yc.repeat(yck_collect.size(0), 1)                                                  # (1, bs) -> (c, bs)
            weights = (yc - yck_collect)/(yc + 1e-10)                                               # (c, bs)
            weights = weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)          # (bs, c, h, w)
            localization_map = (weights * pooler_out).sum(dim=1)                                    # (bs, h, w)

        elif method == "spatial":                                                       # Spatial location wise ablation to check
            for i in range(pooler_out.size(2)):
                for j in range(pooler_out.size(3)):
                    ablation_map = pooler_out.detach().clone()
                    ablation_map[:, :, i, j] = 0.
                    with torch.no_grad():
                        if class_activation == "sigmoid":
                            changed_probs = torch.sigmoid(self.classifier(ablation_map))
                        elif class_activation == "softmax":
                            changed_probs = F.softmax(self.classifier(ablation_map), dim=-1)
                    yck_collect.append(torch.gather(changed_probs, 1, trg.view(-1, 1)).contiguous().view(1, -1))

            # Compute weights and activated localization map
            yck_collect = torch.cat(yck_collect, dim=0)                                             # (h*w, bs) 
            yc = yc.repeat(yck_collect.size(0), 1)                                                  # (1, bs) -> (h*w, bs)
            weights = (yc - yck_collect)/(yc + 1e-10)                                               # (h*w, bs)
            weights = weights.permute(1, 0).view(bs, 1, h, w).repeat(1, c, 1, 1)                    # (bs, c, h, w)
            localization_map = (weights * pooler_out).sum(dim=1)                                    # (bs, h, w)

        if final_activation == "relu":
            localization_map = F.relu(localization_map.view(bs, -1)).view(bs, h, w)
        elif final_activation == "softmax":
            localization_map = F.softmax(localization_map.view(bs, -1), dim=1).view(bs, h, w)
        else:
            raise ValueError(f"Unrecognized activation method {final_activation}")

        localization_map = localization_map.cpu().numpy()
        return localization_map, true_names, pred_names


    def superimpose_image_with_map(self, normal_batch, localization_maps, true_names, pred_names):
        if "imagenet" in self.args["data_root"]:
            _, img, trg = normal_batch 
        elif "cifar" in self.args["data_root"]:
            img, trg = normal_batch

        img = img.permute(0, 2, 3, 1).cpu().numpy()
        bs, h, w, c = img.shape 
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]

        blended_images = []
        for i in range(bs):
            image, loc_map = img[i], (localization_maps[i] * 255).astype(np.uint8)
            loc_map = cv2.resize(loc_map, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap = jet_colors[loc_map]
        
            image, heatmap = (image * 255).astype(np.uint8), (heatmap * 255).astype(np.uint8)
            image, heatmap = Image.fromarray(image).convert("RGBA"), Image.fromarray(heatmap).convert("RGBA")
            blend = Image.blend(image, heatmap, 0.4)
            blended_images.append(blend)

        tilesize = np.ceil(np.sqrt(len(blended_images))).astype('int')
        fig = plt.figure(figsize=(15, 15))
        for i in range(len(blended_images)):
            ax = fig.add_subplot(tilesize, tilesize, i+1)
            image = blended_images[i]
            ax.imshow(image)
            ax.set_title(f"True: {true_names[i]}\nPred: [{round(pred_names[i][1], 2)}] {pred_names[i][0]}", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"./outputs/{self.args['model']}_{self.method}_{self.clf_act_fn}_{dt.now().strftime('%H-%M_%d-%m-%Y')}.png")


    def run_for_batches(self, num_batches=1):
        print("\n[INFO] Started generating plots...")
        
        if "imagenet" in self.args["data_root"]:
            for idx, batch in enumerate(self.network_loader):
                localization_maps, true_names, pred_names = self.generate_localization_map(
                    batch, self.method, self.act_fn, self.clf_act_fn)
                self.superimpose_image_with_map(batch, localization_maps, true_names, pred_names)
                
                if (idx + 1) >= num_batches:
                    print(f"[INFO] Completed {num_batches} batches! Exiting...")
                    break

        elif "cifar" in self.args["data_root"]:
            for idx, (network_batch, normal_batch) in enumerate(zip(self.network_loader, self.normal_loader)):
                localization_maps, true_names, pred_names = self.generate_localization_map(
                    network_batch, self.method, self.act_fn, self.clf_act_fn)
                self.superimpose_image_with_map(normal_batch, localization_maps, true_names, pred_names)
                
                if (idx + 1) >= num_batches:
                    print(f"[INFO] Completed {num_batches} batches! Exiting...")
                    break