from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(results: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        steps = [entry["step"] for entry in result["train_history"]]
        losses = [entry["loss"] for entry in result["train_history"]]
        plt.plot(steps, losses, label=f"{name} train")
    plt.xlabel("Gradient step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss versus gradient step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_validation_curves(results: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        steps = [entry["step"] for entry in result["valid_history"]]
        losses = [entry["loss"] for entry in result["valid_history"]]
        plt.plot(steps, losses, marker="o", label=name)
    plt.xlabel("Gradient step")
    plt.ylabel("Validation loss")
    plt.title("Validation loss comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_perplexity_curves(results: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        steps = [entry["step"] for entry in result["valid_history"]]
        losses = [entry["prplexity"] for entry in result["valid_history"]]
        plt.plot(steps, losses, marker="o", label=name)
    plt.xlabel("Gradient step")
    plt.ylabel("Perplexity")
    plt.title("Perplexity comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
