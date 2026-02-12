import pandas as pd
import matplotlib.pyplot as plt


def main():

    # Load final analysis CSV
    df = pd.read_csv("final_efficiency_analysis.csv")

    # Sort by timestep
    df = df.sort_values("Timestep")

    T = df["Timestep"]

    # -----------------------------
    # Plot 1: Accuracy vs T
    # -----------------------------
    plt.figure()
    plt.plot(T, df["Accuracy (%)"])
    plt.xlabel("Timestep (T)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Timestep")
    plt.savefig("accuracy_vs_T.png")
    plt.show()

    # -----------------------------
    # Plot 2: Inference Time vs T
    # -----------------------------
    plt.figure()
    plt.plot(T, df["Total Inference Time (s)"])
    plt.xlabel("Timestep (T)")
    plt.ylabel("Total Inference Time (s)")
    plt.title("Inference Time vs Timestep")
    plt.savefig("time_vs_T.png")
    plt.show()

    # -----------------------------
    # Plot 3: Spikes vs T
    # -----------------------------
    plt.figure()
    plt.plot(T, df["Average Spikes per Image"])
    plt.xlabel("Timestep (T)")
    plt.ylabel("Average Spikes per Image")
    plt.title("Spike Count vs Timestep")
    plt.savefig("spikes_vs_T.png")
    plt.show()

    # -----------------------------
    # Plot 4: Efficiency Score vs T
    # -----------------------------
    plt.figure()
    plt.plot(T, df["Efficiency Score"])
    plt.xlabel("Timestep (T)")
    plt.ylabel("Efficiency Score")
    plt.title("Efficiency Score vs Timestep")
    plt.savefig("efficiency_vs_T.png")
    plt.show()

    print("\nAll plots saved successfully:")
    print(" - accuracy_vs_T.png")
    print(" - time_vs_T.png")
    print(" - spikes_vs_T.png")
    print(" - efficiency_vs_T.png\n")


if __name__ == "__main__":
    main()
