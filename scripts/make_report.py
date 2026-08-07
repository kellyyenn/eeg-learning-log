"""Turn results/*.csv into a leaderboard, figures, and a Markdown results section."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

RESULTS = Path("results")
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Protocol ranked by how trustworthy the number is. The first one present in the
# data becomes the headline.
PROTOCOL_RANK = [
    "cross-subject (LOSO)",
    "cross-session (T->E, official)",
    "within-subject",
]
TAG = sys.argv[1] if len(sys.argv) > 1 else "iv2a"


def load_all() -> pd.DataFrame:
    frames = []
    for stem, fam in [("classical_per_subject", "classical"),
                      ("crosssession_per_subject", "classical"),
                      ("deep_per_subject", "deep")]:
        p = RESULTS / f"{stem}_{TAG}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["family"] = fam
            frames.append(df)
    if not frames:
        sys.exit(f"No results for tag {TAG!r}. Run scripts/run_benchmark.py first.")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["model", "protocol", "subject"], keep="last")


# A protocol only earns the headline if enough subjects back it. LOSO trained on
# a single subject is degenerate -- it scores at chance for reasons that have
# nothing to do with the models -- so it needs a real pool before it can lead.
MIN_SUBJECTS = {"cross-subject (LOSO)": 4}


def pick_headline(df) -> str:
    present = {p: df[df.protocol == p].subject.nunique() for p in df.protocol.unique()}
    for p in PROTOCOL_RANK:
        if p in present and present[p] >= MIN_SUBJECTS.get(p, 1):
            return p
    return max(present, key=present.get)


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["model", "family", "protocol"])
        .agg(kappa_mean=("kappa", "mean"), kappa_std=("kappa", "std"),
             bacc_mean=("balanced_accuracy", "mean"), bacc_std=("balanced_accuracy", "std"),
             n=("subject", "count"))
        .reset_index()
    )
    g["kappa"] = g.apply(lambda r: f"{r.kappa_mean:.3f} ± {r.kappa_std:.3f}", axis=1)
    g["balanced_acc"] = g.apply(lambda r: f"{r.bacc_mean:.3f} ± {r.bacc_std:.3f}", axis=1)
    return g.sort_values(["protocol", "kappa_mean"], ascending=[True, False])


def fig_per_subject(df, protocol):
    sub = df[df.protocol == protocol]
    if sub.empty:
        return
    piv = sub.pivot_table(index="subject", columns="model", values="kappa")
    order = piv.mean().sort_values(ascending=False).index
    piv = piv[order]
    ax = piv.plot(kind="bar", figsize=(12, 5), width=0.82)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Cohen's kappa")
    ax.set_xlabel("Subject")
    ax.set_title(f"Per-subject decoding performance — {protocol}\nBCI IV-2a, 4-class motor imagery (chance kappa = 0)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / f"per_subject_kappa_{TAG}.png", dpi=160)
    plt.close()


def fig_protocol_gap(df, HEADLINE):
    g = df.groupby(["model", "protocol"]).kappa.mean().unstack()
    if g.shape[1] < 2:
        return
    g = g.sort_values(HEADLINE, ascending=False)
    ax = g.plot(kind="barh", figsize=(9, 5))
    ax.set_xlabel("Cohen's kappa (mean over subjects)")
    ax.set_title("Protocol decides the number, not just the model\n"
                 "shuffled within-subject CV flatters; a held-out session or subject does not")
    plt.tight_layout()
    plt.savefig(FIGS / f"protocol_gap_{TAG}.png", dpi=160)
    plt.close()


def fig_confusion(best_model: str):
    for f in [f"crosssession_preds_{TAG}.npz", f"classical_loso_preds_{TAG}.npz",
              f"deep_loso_preds_{TAG}.npz"]:
        p = RESULTS / f
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        if best_model not in z:
            continue
        y_true, y_pred = z["y_true"], z[best_model]
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{best_model} — held-out confusion matrix\n(row-normalized; diagonal = recall)", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(FIGS / f"confusion_best_{TAG}.png", dpi=160)
        plt.close()
        return


def stats_table(df, HEADLINE) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank across the 9 subjects, best model vs each other."""
    sub = df[df.protocol == HEADLINE]
    piv = sub.pivot_table(index="subject", columns="model", values="kappa")
    best = piv.mean().idxmax()
    rows = []
    for m in piv.columns:
        if m == best:
            continue
        a, b = piv[best].dropna(), piv[m].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) < 5:
            continue
        try:
            stat, p = wilcoxon(a[idx], b[idx])
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(dict(best=best, versus=m, n=len(idx),
                         delta_kappa=float(a[idx].mean() - b[idx].mean()),
                         wilcoxon_stat=stat, p_value=p))
    if not rows:  # fewer than 5 subjects -- a paired test would be meaningless
        return pd.DataFrame(columns=["best", "versus", "n", "delta_kappa",
                                     "wilcoxon_stat", "p_value", "p_holm"])
    out = pd.DataFrame(rows).sort_values("p_value")
    k = len(out)  # Holm correction over the family of comparisons
    out["p_holm"] = np.minimum.accumulate(
        np.minimum(1.0, out.p_value.values * (k - np.arange(k)))[::-1]
    )[::-1]
    return out.sort_values("delta_kappa")


def main():
    df = load_all()
    HEADLINE = pick_headline(df)
    print(f"[report] tag={TAG} headline protocol = {HEADLINE}")
    lb = leaderboard(df)
    lb.to_csv(RESULTS / f"leaderboard_{TAG}.csv", index=False)

    head = lb[lb.protocol == HEADLINE]
    best = head.iloc[0].model if not head.empty else lb.iloc[0].model

    fig_per_subject(df, HEADLINE)
    fig_protocol_gap(df, HEADLINE)
    fig_confusion(best)
    st = stats_table(df, HEADLINE)
    st.to_csv(RESULTS / f"model_comparison_stats_{TAG}.csv", index=False)

    # ---- markdown ----
    n_subj = df.subject.nunique()
    subj_list = ", ".join(f"S{s}" for s in sorted(df.subject.unique()))
    lines = ["## Results\n",
             "BCI Competition IV-2a (`BNCI2014_001`), 22 EEG channels, 4-class motor "
             "imagery (left hand / right hand / feet / tongue), 8–30 Hz band-pass, seed 42.",
             f"**Headline protocol: {HEADLINE}.** Chance kappa = 0.\n"]
    if n_subj < 9:
        lines += [
            f"> **Caveat — {n_subj} of 9 subjects ({subj_list}).** The dataset's only "
            "host, `lampx.tugraz.at`, accepts the TCP connection on :443 and then drops "
            "the TLS handshake; `bnci-horizon-2020.eu` just 302s to the same machine. "
            "Only these subjects were already cached. `scripts/fetch_iv2a.py --watch` "
            "polls for the rest and the full run is one command once they land. Every "
            "number below is real but rests on a small sample — treat the per-subject "
            "spread, not the mean, as the honest summary.\n",
        ]
    for proto in [HEADLINE] + [p for p in PROTOCOL_RANK if p != HEADLINE]:
        part = lb[lb.protocol == proto]
        if part.empty:
            continue
        lines += [f"### {proto}\n",
                  "| model | family | Cohen's kappa | balanced acc | n subj |",
                  "|---|---|---|---|---|"]
        for _, r in part.iterrows():
            lines.append(f"| {r.model} | {r.family} | {r.kappa} | {r.balanced_acc} | {int(r.n)} |")
        lines.append("")
    lk = RESULTS / f"leakage_demo_{TAG}.json"
    if lk.exists():
        j = json.loads(lk.read_text())
        lines += ["### The leakage demo\n",
                  f"Fitting CSP on **all** data before splitting inflates balanced accuracy from "
                  f"**{j['honest_balanced_accuracy']:.3f}** (honest, fit in-fold) to "
                  f"**{j['leaky_balanced_accuracy']:.3f}** — a free "
                  f"**+{j['inflation']:.3f}** that is entirely an artifact of peeking.\n"]
    if not st.empty:
        lines += [f"### Is `{best}` significantly better?\n",
                  "Paired Wilcoxon signed-rank across the 9 subjects, Holm-corrected. "
                  "With n=9 the test has low power — treat non-significant gaps as *undecided*, not equal.\n",
                  "| vs | Δ kappa | p | p (Holm) |", "|---|---|---|---|"]
        for _, r in st.iterrows():
            lines.append(f"| {r.versus} | {r.delta_kappa:+.3f} | {r.p_value:.4f} | {r.p_holm:.4f} |")
        lines.append("")
    lines += ["### Figures\n",
              f"![per-subject](figures/per_subject_kappa_{TAG}.png)",
              f"![protocol gap](figures/protocol_gap_{TAG}.png)",
              f"![confusion](figures/confusion_best_{TAG}.png)"]
    (RESULTS / f"RESULTS_{TAG}.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\n-> results/RESULTS_{TAG}.md, results/leaderboard.csv, results/figures/*.png")


if __name__ == "__main__":
    main()
