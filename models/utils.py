import glob
import os
import clip
import torch
import torchvision.models as models

def build_model(args):
    model_type = args.model
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
    if args.load == "True":
        model = load_model(args, model)
    return model

"""def load_model(args, net):
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
        break"""


def load_model(args, model):
    """Properly load CLIP model components"""
    pattern = os.path.join("./data", f"{args.dataset}_{args.model}_clip0.pth")
    matching_files = glob.glob(pattern)

    if not matching_files:
        print(f"No CLIP checkpoint found matching {pattern}")
        return model

    latest_checkpoint = max(matching_files, key=os.path.getctime)

    try:
        state_dict = torch.load(latest_checkpoint, map_location="cpu")

        # Load visual encoder
        if 'visual_state_dict' in state_dict:
            model.visual.load_state_dict(state_dict['visual_state_dict'])

        # Load text projection (parameter)
        if 'text_projection' in state_dict:
            model.text_projection.data = state_dict['text_projection']

        # Load logit scale
        if 'logit_scale' in state_dict:
            model.logit_scale.data = state_dict['logit_scale']

        # Load text encoder if exists
        if hasattr(model, 'transformer') and 'text_state_dict' in state_dict:
            model.transformer.load_state_dict(state_dict['text_state_dict'])

        print(f"Loaded CLIP model from {latest_checkpoint}")
        return model

    except Exception as e:
        print(f"Error loading CLIP model: {str(e)}")
        return model


def save_model(args, model):
    """Properly save CLIP model components"""
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare state dictionary for CLIP
    state_dict = {
        'visual_state_dict': model.visual.state_dict(),
        'text_projection': model.text_projection.data,  # Save as tensor
        'logit_scale': model.logit_scale.data,
        'args': vars(args)
    }

    # Handle text encoder if exists
    if hasattr(model, 'transformer'):
        state_dict['text_state_dict'] = model.transformer.state_dict()

    version = 0
    while True:
        save_path = f"./data/{args.dataset}_{args.model}_clip{version}.pth"
        if not os.path.exists(save_path):
            break
        version += 1

    torch.save(state_dict, save_path)
    print(f"Saved CLIP model to {save_path}")

"""import os
from datetime import datetime

def save_model(args, net):
    # Create directory if needed
    os.makedirs("./data", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./data/{args.dataset}_{args.model}_{timestamp}_clip.pth"
    
    # Save full model (not just state_dict) to preserve tokenizer/config
    torch.save({
        'model_state_dict': net.state_dict(),
        'model_config': getattr(net, 'config', {}),  # If available
        'args': vars(args)  # Save all arguments
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return save_path"""