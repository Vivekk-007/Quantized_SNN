import pandas as pd


def main():

    # Load previous results
    timestep_df = pd.read_csv("timestep_study_results.csv")
    spike_df = pd.read_csv("spike_analysis_results.csv")

    # Merge on Timestep
    final_df = pd.merge(timestep_df, spike_df, on="Timestep")

    # Energy Proxy (spikes per image)
    final_df["Energy Proxy"] = final_df["Average Spikes per Image"]

    # Efficiency Score = Accuracy / Energy
    final_df["Efficiency Score"] = (
        final_df["Accuracy (%)"] / final_df["Energy Proxy"]
    )

    # Sort by Efficiency Score (higher is better)
    final_df = final_df.sort_values(by="Efficiency Score", ascending=False)

    # Save
    final_df.to_csv("final_efficiency_analysis.csv", index=False)

    print("\n=======================================")
    print("FINAL EFFICIENCY ANALYSIS")
    print(final_df)
    print("=======================================\n")

    print("Results saved to final_efficiency_analysis.csv\n")


if __name__ == "__main__":
    main()
