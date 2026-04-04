import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =========================================================
# Settings you can change directly
# =========================================================

results_dir = "comparison_results"

groups_to_plot = [
    "MLP Architecture",
    "Pretraining Branches",
    "Learning Rate",
    "Finetune LR",
    "GMF latent dimension",
    "Negative Sampling Ratio",
]

plot_train_loss_in_full_curve = True
plot_val_loss_in_full_curve = True
plot_only_neumf_for_group_comparison = True


# =========================================================
# Make plots
# =========================================================

for group in groups_to_plot:
    group_dir = os.path.join(results_dir, group)

    run_names = sorted(os.listdir(group_dir))

    plt.figure(figsize=(8, 5))

    for run_name in run_names:
        run_dir = os.path.join(group_dir, run_name)
        npz_path = os.path.join(run_dir, "history_and_results.npz")

        data = np.load(npz_path, allow_pickle=True)

        stage = data["stage"]
        epoch = data["epoch"]
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]
        config = json.loads(str(data["config_json"]))

        # -------------------------------------------------
        # Save full curve for this run
        # -------------------------------------------------

        x = np.arange(1, len(stage) + 1)

        plt.figure(figsize=(8, 5))

        if plot_train_loss_in_full_curve:
            plt.plot(x, train_loss, label="train loss")

        if plot_val_loss_in_full_curve:
            plt.plot(x, val_loss, label="val loss")

        for i in range(1, len(stage)):
            if stage[i] != stage[i - 1]:
                plt.axvline(i + 0.5, linestyle="--")

        plt.xlabel("Training step across all stages")
        plt.ylabel("Loss")
        plt.title(run_name)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "full_training_curve.png"))
        plt.close()

        # -------------------------------------------------
        # Add line to group comparison plot
        # -------------------------------------------------

        if plot_only_neumf_for_group_comparison:
            neumf_mask = stage == "neumf_train"
            x_plot = epoch[neumf_mask]
            y_plot = val_loss[neumf_mask]
        else:
            x_plot = x
            y_plot = val_loss

        plt.plot(x_plot, y_plot, label=run_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(group)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, "group_comparison_val_loss.png"))
    plt.close()

print("Done.")