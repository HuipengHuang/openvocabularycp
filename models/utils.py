import os
import clip
import torch
import torchvision.models as models

def build_model(model_type):
    if model_type == "resnet50":
        model, preprocess = clip.load("RN50")
    elif model_type == "resnet101":
        model, preprocess = clip.load("RN101")
    elif model_type == "RN50x4":
        model, preprocess = clip.load("RN50x4")
    elif model_type == "ViT-B/16":
        model, preprocess = clip.load("ViT-B/16")
    elif model_type == "ViT-B/32":
        model, preprocess = clip.load("ViT-B/32")
    elif model_type == "ViT-L/14@336px":
        model, preprocess = clip.load("ViT-L/14@336px")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def load_model(args, net):
        p = f"./data/{args.dataset}_{args.model}{0}clip.pth"

        if args.model == "resnet50":
            net.resnet.load_state_dict(torch.load(p))
        else:
            net.load_state_dict(torch.load(p))

def save_model(args, net):
    i = 0
    while (True):
        p = f"./data/{args.dataset}_{args.model}{i}clip.pth"

        if os.path.exists(p):
            i += 1
            continue
        torch.save(net.state_dict(), p)
        break

