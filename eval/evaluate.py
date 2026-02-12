import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from data.load_dataset import load_data
from models.snn_model import SpikingCNN
from models.ann_model import ANN_CNN
from utils.checkpoint import load_checkpoint


def evaluate(model_type="snn"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    if model_type == "snn":
        model = SpikingCNN(T=10, slope=10).to(device)
        model = load_checkpoint(model, "best_model.pth", device)
        print("Loaded Spiking CNN model\n")

    elif model_type == "ann":
        model = ANN_CNN().to(device)
        model = load_checkpoint(model, "best_ann_model.pth", device)
        print("Loaded ANN CNN model\n")

    else:
        raise ValueError("model_type must be 'snn' or 'ann'")

    model.eval()

    _, test_loader = load_data(batch_size=64)

    correct = 0
    total = 0

    total_time = 0
    total_batches = 0


    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_batches += 1

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_batch_time = total_time / total_batches
    avg_image_time = total_time / total


    print("====================================")
    print(f"Model Type: {model_type.upper()}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Inference Time: {total_time:.4f} seconds")
    print(f"Average Time per Batch: {avg_batch_time:.6f} seconds")
    print(f"Average Time per Image: {avg_image_time:.8f} seconds")
    print("====================================\n")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="snn",
        choices=["snn", "ann"],
        help="Choose model type"
    )

    args = parser.parse_args()

    evaluate(model_type=args.model)
