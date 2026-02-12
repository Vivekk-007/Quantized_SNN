import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import pandas as pd

from data.load_dataset import load_data
from models.snn_model import SpikingCNN
from utils.checkpoint import load_checkpoint


def evaluate_model(T_value, device):

    model = SpikingCNN(T=T_value, slope=10).to(device)
    model = load_checkpoint(model, "best_model.pth", device)

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

            total_time += (end_time - start_time)
            total_batches += 1

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_batch_time = total_time / total_batches

    return accuracy, total_time, avg_batch_time


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    timestep_values = [1, 5, 10, 20]

    results = []

    for T in timestep_values:

        print(f"Running evaluation for T = {T}")

        acc, total_time, avg_batch_time = evaluate_model(T, device)

        results.append({
            "Timestep": T,
            "Accuracy (%)": acc,
            "Total Inference Time (s)": total_time,
            "Avg Batch Time (s)": avg_batch_time
        })

        print(f"T={T} | Acc={acc:.2f}% | Time={total_time:.2f}s\n")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("timestep_study_results.csv", index=False)

    print("=======================================")
    print("FINAL RESULTS")
    print(df)
    print("=======================================")
    print("\nResults saved to timestep_study_results.csv\n")


if __name__ == "__main__":
    main()
