import argparse
from pathlib import Path

from data_utils import load_and_train_model
from simulator import run_all_experiments
from plot_results import (
    save_results_csv,
    plot_accuracy_vs_budget,
    plot_avg_aoi_bar,
    write_summary_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Freshness-aware semantic scheduling pilot experiment")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--episode-len", type=int, default=5, help="Number of slots per episode")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4], help="Bandwidth budgets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["random", "aoi", "semantic", "joint"]

    clf, X_test, y_test = load_and_train_model(seed=args.seed)
    results = run_all_experiments(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        budgets=args.budgets,
        methods=methods,
        n_episodes=args.episodes,
        episode_len=args.episode_len,
        seed=args.seed,
    )

    save_results_csv(results, args.budgets, methods, output_dir / "results.csv")
    plot_accuracy_vs_budget(results, args.budgets, methods, output_dir / "accuracy_vs_budget.png")
    fixed_budget_index = 1 if len(args.budgets) > 1 else 0
    plot_avg_aoi_bar(results, args.budgets, methods, fixed_budget_index, output_dir / f"avg_aoi_bar_budget{args.budgets[fixed_budget_index]}.png")
    write_summary_table(results, args.budgets, methods, fixed_budget_index, output_dir / f"summary_budget{args.budgets[fixed_budget_index]}.txt")

    print("Done. Files saved to:", output_dir.resolve())
    print("- results.csv")
    print("- accuracy_vs_budget.png")
    print(f"- avg_aoi_bar_budget{args.budgets[fixed_budget_index]}.png")
    print(f"- summary_budget{args.budgets[fixed_budget_index]}.txt")


if __name__ == "__main__":
    main()
