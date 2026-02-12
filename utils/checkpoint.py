import torch


def save_checkpoint(model, path="model.pth"):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path="model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
