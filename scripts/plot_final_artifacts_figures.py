from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#1f2937",
        "axes.titlecolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_artifacts"
FIGURES = ROOT / "figures" / "final_artifacts"
FIGURES.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "cosine": "Cosine",
    "history_cosine": "History-Cosine",
    "trustfl_chain": "TrustFL-Chain",
}

METHOD_COLORS = {
    "FedAvg": "#9aa3ad",
    "Trimmed Mean": "#86a4b5",
    "Cosine": "#a5aa82",
    "History-Cosine": "#b5a0aa",
    "TrustFL-Chain": "#7893aa",
}

MUTED_PAIR = ["#86a4b5", "#a5aa82"]
LINE_COLORS = ["#7893aa", "#a5aa82", "#b5a0aa", "#9aa3ad"]
DETECTOR_METHODS = {"trustfl_chain", "history_cosine"}


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def save_figure(path: str) -> None:
    png_path = FIGURES / path
    plt.savefig(png_path, dpi=220)
    plt.savefig(png_path.with_suffix(".pdf"))
    plt.savefig(png_path.with_suffix(".svg"))


def save_bar(df, x, y, title, ylabel, path, color="#3b82f6"):
    if df.empty:
        return
    plt.figure(figsize=(8, 4.2))
    plt.bar(df[x], df[y], color=color, edgecolor="#334155", linewidth=0.4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save_figure(path)
    plt.close()


def save_grouped(df, category, method, value, title, ylabel, path):
    if df.empty:
        return
    pivot = df.pivot_table(index=category, columns=method, values=value, aggfunc="mean")
    order = [m for m in METHOD_LABELS.values() if m in pivot.columns]
    pivot = pivot[order] if order else pivot
    colors = [METHOD_COLORS.get(col, "#7c8491") for col in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.4), width=0.78, color=colors, edgecolor="#334155", linewidth=0.35)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.8)
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure(path)
    plt.close()


def grouped_bar_with_error_and_points(
    agg_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    category: str,
    method: str,
    value_mean: str,
    value_std: str,
    seed_value: str,
    title: str,
    ylabel: str,
    path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    if agg_df.empty:
        return
    categories = list(dict.fromkeys(agg_df[category].tolist()))
    methods = [m for m in METHOD_LABELS.values() if m in set(agg_df[method])]
    x = np.arange(len(categories), dtype=float)
    width = min(0.78 / max(len(methods), 1), 0.18)
    fig, ax = plt.subplots(figsize=(9.2, 4.7))

    for idx, method_name in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        means, stds = [], []
        for cat in categories:
            row = agg_df[(agg_df[category] == cat) & (agg_df[method] == method_name)]
            means.append(float(row[value_mean].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row[value_std].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=2.5,
            color=METHOD_COLORS.get(method_name, "#9aa3ad"),
            edgecolor="#334155",
            linewidth=0.35,
            error_kw={"elinewidth": 0.8, "ecolor": "#475569"},
            label=method_name,
        )
        for cat_idx, cat in enumerate(categories):
            pts = seed_df[(seed_df[category] == cat) & (seed_df[method] == method_name)][seed_value].dropna().tolist()
            if not pts:
                continue
            jitter = np.linspace(-width * 0.18, width * 0.18, len(pts)) if len(pts) > 1 else [0]
            ax.scatter(
                np.full(len(pts), x[cat_idx] + offset) + jitter,
                pts,
                s=13,
                color="#111827",
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.75)
    ax.legend(fontsize=8, ncol=2)
    ax.text(0.99, 0.02, "bars: mean +/- SD; dots: seeds", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#64748b")
    plt.tight_layout()
    save_figure(path)
    plt.close()


def architecture():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    boxes = [
        ("Clients\nlocal SGD", 0.05, 0.55),
        ("Poisoned / benign\nupdates", 0.22, 0.55),
        ("Feature extractor\nnorm, cosine, median,\ntemporal, reputation", 0.43, 0.55),
        ("Risk detector +\ndynamic reputation", 0.66, 0.55),
        ("Weighted\naggregation", 0.84, 0.55),
        ("Hash-chain audit\nledger", 0.58, 0.12),
    ]
    for text, x, y in boxes:
        ax.add_patch(Rectangle((x, y), 0.14, 0.22, fill=True, facecolor="#f8fafc", linewidth=1.2, edgecolor="#64748b"))
        ax.text(x + 0.07, y + 0.11, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.19, 0.66), (0.22, 0.66)),
        ((0.36, 0.66), (0.43, 0.66)),
        ((0.57, 0.66), (0.66, 0.66)),
        ((0.80, 0.66), (0.84, 0.66)),
        ((0.50, 0.55), (0.62, 0.34)),
        ((0.73, 0.55), (0.67, 0.34)),
        ((0.71, 0.55), (0.89, 0.55)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.1, color="#64748b"))
    plt.tight_layout()
    save_figure("system_architecture.png")
    plt.close()


def blockchain_audit_concept():
    fig, ax = plt.subplots(figsize=(3.55, 3.20))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def rounded_box(x, y, w, h, label, face, edge="#6b7f93", size=8.0, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, fontweight=weight, color="#1f2937")

    def arrow(start, end, color="#708090", lw=1.15, scale=11, alpha=1.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=scale, linewidth=lw, color=color, alpha=alpha))

    def ledger_block(x, y, idx, face):
        w, h = 0.19, 0.20
        ax.add_patch(Rectangle((x + 0.010, y - 0.010), w, h, facecolor="#d9e2e8", edgecolor="none", alpha=0.55))
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="#53677a", linewidth=1.05))
        ax.text(x + 0.014, y + h - 0.038, f"B{idx}", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#1f2937")
        ax.text(x + w - 0.014, y + h - 0.038, f"hash h{idx}", ha="right", va="center", fontsize=5.4, color="#53677a")
        fields = ["prev " + ("GEN" if idx == 1 else f"h{idx-1}"), "update h(u)", "risk q / rep r"]
        for k, text in enumerate(fields):
            yy = y + h - 0.082 - 0.043 * k
            ax.plot([x + 0.012, x + w - 0.012], [yy + 0.020, yy + 0.020], color="#d7dee4", lw=0.55)
            ax.text(x + 0.014, yy, text, ha="left", va="center", fontsize=5.9, color="#475569")
        return x + w, y + h / 2

    ax.text(0.50, 0.96, "Update decisions committed to a hash-linked ledger", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#111827")

    client_positions = [(0.04, 0.73), (0.04, 0.57), (0.04, 0.41)]
    for idx, (x, y) in enumerate(client_positions, start=1):
        rounded_box(x, y, 0.18, 0.09, f"C{idx}\nupdate", "#f7fafc", edge="#8fa2b3", size=5.9)

    blocks = [
        ("Score", 0.33, 0.67, "#edf3f6"),
        ("Rep.\nweight", 0.51, 0.67, "#f0f4ec"),
        ("Aggregate", 0.70, 0.67, "#edf3f7"),
    ]
    for text, x, y, color in blocks:
        rounded_box(x, y, 0.145, 0.105, text, color, edge="#71879a", size=5.9)

    for start_y in [0.775, 0.615, 0.455]:
        arrow((0.22, start_y), (0.34, 0.722), color="#8795a1", lw=0.9, scale=8, alpha=0.9)
    for x1, x2 in [(0.475, 0.51), (0.655, 0.70)]:
        arrow((x1, 0.722), (x2, 0.722), color="#8795a1", lw=0.9, scale=8)

    rounded_box(0.865, 0.665, 0.115, 0.115, "Global\nmodel", "#f7fafc", edge="#8fa2b3", size=5.5)
    arrow((0.845, 0.722), (0.865, 0.722), color="#8795a1", lw=0.9, scale=8)

    block_centres = []
    ledger_x = [0.08, 0.315, 0.55]
    for j, x in enumerate(ledger_x, start=1):
        end_x, mid_y = ledger_block(x, 0.20, j, "#f7f5ed" if j % 2 else "#eef4f4")
        block_centres.append((x + 0.0625, mid_y))
        if j > 1:
            prev_x = ledger_x[j - 2] + 0.19
            arrow((prev_x + 0.012, mid_y), (x - 0.012, mid_y), color="#64748b", lw=1.15, scale=10)
            ax.text((prev_x + x) / 2, mid_y + 0.042, f"h{j-1}", ha="center", va="center", fontsize=5.6, color="#526274")

    commit_points = [(0.41, 0.67), (0.59, 0.67), (0.77, 0.67)]
    target_blocks = [block_centres[0], block_centres[1], block_centres[2]]
    for start, end in zip(commit_points, target_blocks):
        arrow(start, (end[0], end[1] + 0.105), color="#8b9aa6", lw=0.8, scale=7, alpha=0.85)

    rounded_box(0.80, 0.235, 0.16, 0.13, "Audit\nquery", "#f8fafc", edge="#8fa2b3", size=6.0)
    arrow((0.75, 0.300), (0.80, 0.300), color="#64748b", lw=1.0, scale=9)
    ax.text(0.50, 0.095, "append-only evidence: prev hash -> update hash -> risk/rep -> aggregation hash", ha="center", va="center", fontsize=5.25, color="#475569")

    plt.tight_layout(pad=0.45)
    save_figure("blockchain_audit_concept.png")
    plt.close()


def reputation_example(summary):
    reps = list((ROOT / "results" / "reduced_matrix").rglob("*trustfl_chain_model_scaling_noniid_a0.5_mr0.2_full_seed42_reputation.csv"))
    if not reps:
        return
    rep = pd.read_csv(reps[0])
    mean = rep.groupby(["round", "malicious"])["reputation"].mean().reset_index()
    plt.figure(figsize=(7, 4))
    for mal, group in mean.groupby("malicious"):
        color = "#9a7f8f" if mal else "#4f6f8f"
        plt.plot(group["round"], group["reputation"], marker="o", markersize=3, linewidth=1.8, color=color, label="malicious" if mal else "benign")
    plt.xlabel("Round")
    plt.ylabel("Mean reputation")
    plt.title("Reputation evolution example")
    plt.grid(axis="both", alpha=0.8)
    plt.legend()
    plt.tight_layout()
    save_figure("reputation_evolution_example.png")
    plt.close()


def main() -> int:
    summary = pd.read_csv(RESULTS / "summary_all_runs.csv")
    agg = pd.read_csv(RESULTS / "aggregate_mean_std.csv")
    done = summary[summary["complete"].astype(bool)].copy()
    architecture()
    blockchain_audit_concept()
    reputation_example(summary)

    main_alpha = agg[(agg["phase"] == "main") & (agg["dataset"] == "fashion_mnist") & (agg["distribution"] == "alpha_0.5")].copy()
    main_alpha["Method"] = main_alpha["defense"].map(method_label)
    main_alpha["Attack"] = main_alpha["attack"].str.replace("_", " ").str.title()
    seed_main_alpha = done[(done["phase"] == "main") & (done["dataset"] == "fashion_mnist") & (done["distribution"] == "alpha_0.5")].copy()
    seed_main_alpha["Method"] = seed_main_alpha["defense"].map(method_label)
    seed_main_alpha["Attack"] = seed_main_alpha["attack"].str.replace("_", " ").str.title()
    grouped_bar_with_error_and_points(
        main_alpha,
        seed_main_alpha,
        "Attack",
        "Method",
        "final_accuracy_mean",
        "final_accuracy_std",
        "final_accuracy",
        "Fashion-MNIST final accuracy under Non-IID alpha=0.5 (n=3)",
        "Final accuracy",
        "main_accuracy_by_attack.png",
        ylim=(0, 0.86),
    )

    detect_alpha = main_alpha[main_alpha["defense"].isin(DETECTOR_METHODS)].copy()
    seed_detect_alpha = seed_main_alpha[seed_main_alpha["defense"].isin(DETECTOR_METHODS)].copy()
    grouped_bar_with_error_and_points(
        detect_alpha,
        seed_detect_alpha,
        "Attack",
        "Method",
        "mean_f1_mean",
        "mean_f1_std",
        "mean_f1",
        "Detection F1 for detector-based methods (n=3)",
        "Mean detection F1",
        "main_detection_f1_by_attack.png",
        ylim=(0, 0.9),
    )

    ablation = agg[agg["phase"] == "ablation"].copy()
    if not ablation.empty:
        full = main_alpha[(main_alpha["defense"] == "trustfl_chain") & (main_alpha["attack"] == "model_scaling")].copy()
        full["ablation"] = "full"
        ablation = pd.concat([full, ablation], ignore_index=True, sort=False)
        variant_order = ["full", "no_ledger", "no_reputation", "no_temporal", "cosine_only"]
        ablation["Variant"] = pd.Categorical(
            ablation["ablation"].map(lambda x: str(x).replace("_", " ").title()),
            categories=[v.replace("_", " ").title() for v in variant_order],
            ordered=True,
        )
        ablation = ablation.sort_values("Variant")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(
            ablation["Variant"].astype(str),
            ablation["final_accuracy_mean"],
            yerr=ablation["final_accuracy_std"].fillna(0),
            capsize=2.5,
            color=MUTED_PAIR[0],
            edgecolor="#334155",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#475569"},
        )
        axes[0].set_ylabel("Final accuracy")
        axes[0].set_title("Ablation accuracy (n=3)")
        axes[1].bar(
            ablation["Variant"].astype(str),
            ablation["mean_f1_mean"],
            yerr=ablation["mean_f1_std"].fillna(0),
            capsize=2.5,
            color=MUTED_PAIR[1],
            edgecolor="#334155",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#475569"},
        )
        axes[1].set_ylabel("Mean detection F1")
        axes[1].set_title("Ablation detection (n=3)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.8)
        fig.text(0.99, 0.01, "error bars: SD across seeds", ha="right", va="bottom", fontsize=7, color="#64748b")
        plt.tight_layout()
        save_figure("ablation_accuracy_f1.png")
        plt.close()

    ratio = agg[agg["phase"] == "ratio"].copy()
    if not ratio.empty:
        ratio["Method"] = ratio["defense"].map(method_label)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
        for idx, (method, group) in enumerate(ratio.groupby("Method")):
            group = group.sort_values("malicious_ratio")
            color = METHOD_COLORS.get(method, LINE_COLORS[idx % len(LINE_COLORS)])
            axes[0].errorbar(
                group["malicious_ratio"],
                group["final_accuracy_mean"],
                yerr=group["final_accuracy_std"],
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                color=color,
                label=method,
            )
            axes[1].errorbar(
                group["malicious_ratio"],
                group["mean_fpr_mean"],
                yerr=group["mean_fpr_std"],
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                color=color,
                label=method,
            )
        axes[0].set_xlabel("Malicious ratio")
        axes[0].set_ylabel("Final accuracy")
        axes[0].set_title("Accuracy")
        axes[1].set_xlabel("Malicious ratio")
        axes[1].set_ylabel("Mean false positive rate")
        axes[1].set_title("False positives")
        for ax in axes:
            ax.grid(axis="both", alpha=0.8)
        axes[0].legend(fontsize=8)
        fig.suptitle("Malicious-ratio sweep under model scaling (n=3)", y=0.98)
        fig.subplots_adjust(top=0.84)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure("malicious_ratio_trend.png")
        plt.close()

    cifar = agg[(agg["dataset"] == "cifar10") & (agg["distribution"] == "alpha_0.5")].copy()
    if not cifar.empty:
        cifar["Method"] = cifar["defense"].map(method_label)
        cifar["Attack"] = cifar["attack"].str.replace("_", " ").str.title()
        seed_cifar = done[(done["dataset"] == "cifar10") & (done["distribution"] == "alpha_0.5")].copy()
        seed_cifar["Method"] = seed_cifar["defense"].map(method_label)
        seed_cifar["Attack"] = seed_cifar["attack"].str.replace("_", " ").str.title()
        grouped_bar_with_error_and_points(
            cifar,
            seed_cifar,
            "Attack",
            "Method",
            "final_accuracy_mean",
            "final_accuracy_std",
            "final_accuracy",
            "CIFAR-10 supplemental final accuracy, 20 rounds (n=3)",
            "Final accuracy",
            "cifar10_supplement.png",
            ylim=(0, 0.36),
        )

    overhead = main_alpha.groupby("Method", as_index=False).agg(
        runtime=("mean_runtime_sec_mean", "mean"), runtime_std=("mean_runtime_sec_std", "mean"), ledger=("final_ledger_bytes_mean", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(overhead["Method"], overhead["runtime"], yerr=overhead["runtime_std"], capsize=2.5, color="#86a4b5", edgecolor="#334155", linewidth=0.4)
    axes[0].set_ylabel("Runtime per round (s)")
    axes[0].set_title("CPU simulator runtime")
    axes[1].bar(overhead["Method"], overhead["ledger"], color="#a5aa82", edgecolor="#334155", linewidth=0.4)
    axes[1].set_ylabel("Ledger bytes")
    axes[1].set_title("Run log / audit ledger bytes")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.8)
    plt.tight_layout()
    save_figure("runtime_ledger_overhead.png")
    plt.close()

    print(f"Wrote final artifact figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
