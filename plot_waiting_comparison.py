import json
import os
import matplotlib.pyplot as plt

DIRECTORY_PATH = "results"

FIGURES = [
    ("Waiting Time: Single Intersection", "waiting", "Total Waiting Time (s)", "waiting-fixed_vs_learning.png", [
        ("Fixed Timing", "fixed-timing"),
        ("Q-Learning", "q-learning"),
        ("Deep Q-Learning", "dql"),
    ]),
    ("Queue Length: Single Intersection", "queue", "Queue Length (vehicles)", "queue-fixed_vs_learning.png", [
        ("Fixed Timing", "fixed-timing"),
        ("Q-Learning", "q-learning"),
        ("Deep Q-Learning", "dql"),
    ]),
    ("Waiting Time: Multi Intersection", "waiting", "Total Waiting Time (s)", "waiting-multiagent.png", [
        ("Cooperative Q-Learning", "cooperative-qlearning-simple"),
        ("Multi-Agent Q-Learning", "multiagent-qlearning"),
        ("Multi-Agent DQL", "multiagent-dql"),
    ]),
    ("Queue Length: Multi Intersection", "queue", "Queue Length (vehicles)", "queue-multiagent.png", [
        ("Cooperative Q-Learning", "cooperative-qlearning-simple"),
        ("Multi-Agent Q-Learning", "multiagent-qlearning"),
        ("Multi-Agent DQL", "multiagent-dql"),
    ]),
]


def load_metric_dataset(tag, metric):
    path = os.path.join(DIRECTORY_PATH, f"{tag}-{metric}.json")
    if not os.path.exists(path):
        print(f"[skip] Missing data for {tag} ({metric}): {path}")
        return None

    with open(path, "r") as fh:
        data = json.load(fh)

    steps = data.get("steps", [])
    values = data.get("values")
    if values is None:
        values = data.get(metric)
    if values is None:
        # fall back to legacy keys
        legacy_key = "waiting" if metric == "waiting" else "queue"
        values = data.get(legacy_key)
    if not steps or not values:
        print(f"[skip] Empty dataset for {tag} ({metric})")
        return None
    return steps, values


def plot_metric(title, metric_key, ylabel, filename, datasets):
    plt.figure(figsize=(10, 6))
    any_plotted = False

    for label, tag in datasets:
        loaded = load_metric_dataset(tag, metric_key)
        if not loaded:
            continue
        steps, values = loaded
        plt.plot(steps, values, label=label)
        any_plotted = True

    if not any_plotted:
        print(f"No datasets found for figure '{title}'.")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(DIRECTORY_PATH, filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    for title, metric_key, ylabel, filename, datasets in FIGURES:
        plot_metric(title, metric_key, ylabel, filename, datasets)


if __name__ == "__main__":
    main()
