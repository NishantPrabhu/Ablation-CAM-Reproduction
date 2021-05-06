
""" 
Network definitions.

Some standard network architectures have been slightly modified
to obtain inferences specific to Ablation-CAM 
"""

import torch
import torch.nn as nn 
from torchvision import models


def get_vgg16_model(pretrained=True, state_dict=None):
    model = models.vgg16(pretrained=pretrained)
    if state_dict:
        model.classifier[-1] = nn.Linear(4096, 10, bias=True)
        model.load_state_dict(state_dict)
    model_until_pooling = nn.Sequential(*list(model.children())[:-2])
    classifier = nn.Sequential(model.avgpool, nn.Flatten(), model.classifier)
    return model, model_until_pooling, classifier


def get_resnet18_model(pretrained=True, state_dict=None):
    model = models.resnet18(pretrained=pretrained)
    if state_dict:
        model.fc = nn.Linear(512, 10, bias=True)
        model.load_state_dict(state_dict)
    model_until_pooling = nn.Sequential(*list(model.children())[:-2])
    classifier = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return model, model_until_pooling, classifier


def get_resnet50_model(pretrained=True, state_dict=None):
    model = models.resnet50(pretrained=pretrained)
    if state_dict:
        model.fc = nn.Linear(2048, 10, bias=True)
        model.load_state_dict(state_dict)
    model_until_pooling = nn.Sequential(*list(model.children())[:-2])
    classifier = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return model, model_until_pooling, classifier


def get_inception_v3_model(pretrained=True, state_dict=None):
    model = models.inception_v3(pretrained=pretrained)
    if state_dict:
        model.fc = nn.Linear(2048, 10, bias=True)
        model.load_state_dict(state_dict)
    model_until_pooling = nn.Sequential(*list(model.children())[:-3])
    classifier = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return model, model_until_pooling, classifier