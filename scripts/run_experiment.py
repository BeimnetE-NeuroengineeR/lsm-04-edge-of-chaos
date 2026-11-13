
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import generate_input_signal
from src.reservoir import run_reservoir_pair
from src.metrics import compute_memory_profile


def main():

    print("=" * 60)
    print("PROJECT 4: EDGE OF CHAOS")
    print("=" * 60)

    # Experimental configuration.
    N_NEURONS = 200
    DURATION = 8.0
    BIN_MS = 10.0
    MAX_DELAY = 20

    SEEDS = [11, 23, 37]

    WEIGHT_SCALES = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50
    ]

    N_SAMPLES = int(
        DURATION * 1000 / BIN_MS
    )

    results = []

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    total_runs = len(SEEDS) * len(WEIGHT_SCALES)
    run_number = 0

    # Repeat the experiment across independent seeds.
    for seed_value in SEEDS:

        print(f"\nStarting random seed: {seed_value}")

        signal = generate_input_signal(
            n_samples=N_SAMPLES,
            seed=seed_value + 1000
        )

        for weight_scale in WEIGHT_SCALES:

            run_number += 1

            print(
                f"\nExperiment {run_number}/{total_runs}"
            )

            # Simulate matched reservoir pair.
            (
                reservoir_states,
                divergence,
                firing_rate
            ) = run_reservoir_pair(
                signal=signal,
                weight_scale=weight_scale,
                seed_value=seed_value,
                n_neurons=N_NEURONS,
                bin_ms=BIN_MS
            )

            # Evaluate memory performance.
            (
                delays,
                capacities,
                total_mc,
                mean_r2
            ) = compute_memory_profile(
                reservoir_states=reservoir_states,
                input_signal=signal,
                max_delay=MAX_DELAY
            )

            print(
                f"Memory capacity: {total_mc:.4f}"
            )

            print(
                f"Perturbation divergence: {divergence:.4f}"
            )

            print(
                f"Mean firing rate: {firing_rate:.2f} Hz"
            )

            results.append({

                "seed": seed_value,

                "weight_scale": weight_scale,

                "memory_capacity": total_mc,

                "mean_test_r2": mean_r2,

                "perturbation_divergence": divergence,

                "mean_firing_rate_hz": firing_rate

            })

    # Save individual experimental results.
    df = pd.DataFrame(results)

    df.to_csv(
        "results/per_seed_results.csv",
        index=False
    )

    # Aggregate results across random seeds.
    summary = (
        df.groupby("weight_scale")
        .agg(
            mc_mean=("memory_capacity", "mean"),
            mc_std=("memory_capacity", "std"),

            divergence_mean=(
                "perturbation_divergence",
                "mean"
            ),

            divergence_std=(
                "perturbation_divergence",
                "std"
            ),

            firing_mean=(
                "mean_firing_rate_hz",
                "mean"
            ),

            firing_std=(
                "mean_firing_rate_hz",
                "std"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 60)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # MAIN FIGURE: MEMORY CAPACITY VS DYNAMICS
    # --------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(
        summary["weight_scale"],
        summary["mc_mean"],
        yerr=summary["mc_std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Memory capacity"
    )

    ax1.set_xlabel(
        "Recurrent Weight Scale",
        fontsize=12
    )

    ax1.set_ylabel(
        "Test Memory Capacity",
        fontsize=12
    )

    ax1.set_title(
        "Recurrent Coupling, Memory and Network Dynamics",
        fontsize=14
    )

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    # Second axis: perturbation sensitivity.
    ax2 = ax1.twinx()

    ax2.errorbar(
        summary["weight_scale"],
        summary["divergence_mean"],
        yerr=summary["divergence_std"],
        marker="s",
        linestyle="--",
        capsize=4,
        linewidth=2,
        label="Perturbation divergence"
    )

    ax2.set_ylabel(
        "Normalized Perturbation Divergence",
        fontsize=12
    )

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/main_result.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # SECOND FIGURE: MEAN FIRING RATE
    # --------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        summary["weight_scale"],
        summary["firing_mean"],
        yerr=summary["firing_std"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    plt.xlabel("Recurrent Weight Scale")
    plt.ylabel("Mean Firing Rate (Hz)")

    plt.title(
        "Reservoir Activity Across Recurrent Coupling Strengths"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        "figures/firing_rate.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Report the highest observed memory performance.
    best_idx = summary["mc_mean"].idxmax()
    best = summary.loc[best_idx]

    print("\n" + "=" * 60)
    print("HIGHEST OBSERVED MEMORY PERFORMANCE")
    print("=" * 60)

    print(
        f"Weight scale: {best['weight_scale']:.2f}"
    )

    print(
        f"Mean memory capacity: {best['mc_mean']:.4f}"
    )

    print(
        "\nResults saved successfully:"
    )

    print("results/per_seed_results.csv")
    print("results/summary.csv")
    print("figures/main_result.png")
    print("figures/firing_rate.png")

    print("\nExperiment completed.")

    print(
        "Note: Perturbation sensitivity is not a formal "
        "Lyapunov exponent or proof of chaos."
    )


if __name__ == "__main__":
    main()
