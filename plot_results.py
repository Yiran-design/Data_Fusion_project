import csv
from pathlib import Path
import matplotlib.pyplot as plt


def save_results_csv(results, budgets, methods, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "budget", "accuracy", "avg_aoi", "avg_cost"])
        for m in methods:
            for i, b in enumerate(budgets):
                writer.writerow([
                    m,
                    b,
                    results[m]["accuracy"][i],
                    results[m]["avg_aoi"][i],
                    results[m]["avg_cost"][i],
                ])


def plot_accuracy_vs_budget(results, budgets, methods, out_png):
    plt.figure(figsize=(7, 4.5))
    for m in methods:
        plt.plot(budgets, results[m]["accuracy"], marker="o", label=m)
    plt.xlabel("Bandwidth Budget")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Bandwidth Budget")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_avg_aoi_bar(results, budgets, methods, fixed_budget_index, out_png):
    fixed_budget = budgets[fixed_budget_index]
    vals = [results[m]["avg_aoi"][fixed_budget_index] for m in methods]

    plt.figure(figsize=(7, 4.5))
    plt.bar(methods, vals)
    plt.ylabel("Average AoI")
    plt.title(f"Average AoI vs. Method (Budget={fixed_budget})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_summary_table(results, budgets, methods, budget_index, out_txt):
    lines = []
    lines.append(f"=== Summary Table (Budget = {budgets[budget_index]}) ===")
    header = "{:<12s} {:>10s} {:>10s} {:>10s}".format("Method", "Accuracy", "Avg_AoI", "Avg_Cost")
    lines.append(header)
    for m in methods:
        line = "{:<12s} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            m,
            results[m]["accuracy"][budget_index],
            results[m]["avg_aoi"][budget_index],
            results[m]["avg_cost"][budget_index],
        )
        lines.append(line)

    out_path = Path(out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
