import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

from data.load_dataset import load_data
from models.snn_model import SpikingCNN
from utils.checkpoint import load_checkpoint


def analyze_spikes(T_value, device):

    model = SpikingCNN(T=T_value, slope=10).to(device)
    model = load_checkpoint(model, "best_model.pth", device)

    model.eval()
    _, test_loader = load_data(batch_size=64)

    total_spikes = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)

            outputs, spike_count = model(images, return_spike_stats=True)

            total_spikes += spike_count
            total_images += images.size(0)

    avg_spikes_per_image = total_spikes / total_images

    return avg_spikes_per_image


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    timestep_values = [1, 5, 10, 20]

    results = []

    for T in timestep_values:

        print(f"Analyzing spikes for T = {T}")

        avg_spikes = analyze_spikes(T, device)

        results.append({
            "Timestep": T,
            "Average Spikes per Image": avg_spikes
        })

        print(f"T={T} | Avg Spikes/Image = {avg_spikes:.2f}\n")

    df = pd.DataFrame(results)
    df.to_csv("spike_analysis_results.csv", index=False)

    print("=======================================")
    print(df)
    print("=======================================")
    print("\nResults saved to spike_analysis_results.csv\n")


if __name__ == "__main__":
    main()
