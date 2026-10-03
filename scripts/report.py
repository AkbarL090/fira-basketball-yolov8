"""Ringkas hasil training: tabel (csv + markdown), grafik akurasi per epoch, confusion matrix.

  python scripts/report.py --model efficientnet_b0
Membaca results/*/summary.json dan history.csv yang dibuat train.py.
"""
import argparse, csv, glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KELAS = ["backboard", "ball", "rim", "latar"]
MODES = ["feature", "partial", "scratch"]
COL = {"feature": "#0072B2", "partial": "#009E73", "scratch": "#D55E00"}


def ms(v, d=3, pct=False):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    f = 100 if pct else 1
    m, s = np.mean(v) * f, np.std(v) * f
    u = "%" if pct else ""
    return f"{m:.{d}f}{u}" if len(v) == 1 else f"{m:.{d}f} ± {s:.{d}f}{u}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--model", default=None, help="model untuk grafik; default = model dengan mode terbanyak")
    a = ap.parse_args()

    S = [json.load(open(p)) for p in sorted(glob.glob(f"{a.dir}/*/summary.json"))]
    S = [s for s in S if not s.get("smoke")]
    if not S:
        raise SystemExit("tidak ada summary.json (atau hanya hasil smoke)")
    grp = defaultdict(list)
    for s in S:
        grp[(s["model"], s["mode"])].append(s)

    rows = []
    for (m, mode), L in sorted(grp.items(), key=lambda k: (k[0][0], MODES.index(k[0][1]))):
        g = lambda k: [x[k] for x in L]
        rows.append(dict(model=m, mode=mode, n_seed=len(L),
                         val_acc_terbaik=ms(g("best_val_acc"), 1, True), test_acc=ms(g("test_acc"), 1, True),
                         epoch_val_ge_90_pertama=ms(g("epoch_first_val_ge_0_9"), 1),
                         epoch_val_ge_90_stabil=ms(g("epoch_stable_val_ge_0_9"), 1),
                         waktu_latih_s=ms(g("train_time_s"), 0),
                         param_dilatih=f"{L[0]['params_trained']:,}", param_total=f"{L[0]['params_total']:,}",
                         perangkat=L[0]["device"]))
    with open(f"{a.dir}/ringkasan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    head = ["Model", "Mode", "Seed", "Akurasi val terbaik", "Akurasi test", "Epoch val ≥ 90% (pertama / stabil)",
            "Waktu latih (s)", "Param dilatih / total"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append(f"| {r['model']} | {r['mode']} | {r['n_seed']} | {r['val_acc_terbaik']} | {r['test_acc']} | "
                  f"{r['epoch_val_ge_90_pertama']} / {r['epoch_val_ge_90_stabil']} | {r['waktu_latih_s']} | "
                  f"{r['param_dilatih']} / {r['param_total']} |")
    open(f"{a.dir}/ringkasan.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    model = a.model or max({m for m, _ in grp}, key=lambda m: sum(1 for k in grp if k[0] == m))
    modes = [m for m in MODES if (model, m) in grp]

    # --- grafik per epoch
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    for mode in modes:
        H = []
        for s in grp[(model, mode)]:
            rows_h = list(csv.DictReader(open(f"{a.dir}/{model}_{mode}_s{s['seed']}/history.csv")))
            H.append([[float(r["val_acc"]) for r in rows_h], [float(r["val_loss"]) for r in rows_h]])
        H = np.array(H); ep = np.arange(1, H.shape[2] + 1)
        for k in (0, 1):
            for h in H[:, k]:
                ax[k].plot(ep, h, color=COL[mode], alpha=0.25, lw=1)
            ax[k].plot(ep, H[:, k].mean(0), color=COL[mode], lw=2.2, label=mode, marker="o", ms=3)
    ax[0].axhline(0.9, ls="--", c="#888", lw=0.8); ax[0].set_ylim(0, 1.02)
    ax[0].set_title(f"{model}: akurasi validasi per epoch"); ax[0].set_ylabel("akurasi val")
    ax[1].set_title(f"{model}: loss validasi per epoch"); ax[1].set_ylabel("loss val")
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=.25); x.legend()
    n = len(grp[(model, modes[0])])
    fig.text(0.5, -0.02, f"garis tebal = rata-rata {n} seed, garis tipis = tiap seed; putus-putus = 90%",
             ha="center", fontsize=8, color="#555")
    plt.tight_layout(); plt.savefig(f"{a.dir}/akurasi_per_epoch_{model}.png", bbox_inches="tight"); plt.close()

    # --- confusion matrix test (dijumlah lintas seed, dinormalisasi per baris)
    fig, ax = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 4), dpi=140)
    ax = np.atleast_1d(ax)
    for x, mode in zip(ax, modes):
        C = sum(np.loadtxt(f"{a.dir}/{model}_{mode}_s{s['seed']}/confusion_test.csv", delimiter=",", skiprows=1, dtype=int)
                for s in grp[(model, mode)])
        P = C / C.sum(1, keepdims=True)
        x.imshow(P, vmin=0, vmax=1, cmap="Blues")
        x.set_xticks(range(4), KELAS, rotation=30); x.set_yticks(range(4), KELAS)
        for i in range(4):
            for j in range(4):
                x.text(j, i, f"{C[i, j]}", ha="center", va="center", color="white" if P[i, j] > .5 else "#222", fontsize=9)
        x.set_title(f"{mode} (test)"); x.set_xlabel("prediksi"); x.set_ylabel("sebenarnya")
    plt.tight_layout(); plt.savefig(f"{a.dir}/confusion_{model}.png", bbox_inches="tight"); plt.close()
    print("\ngrafik:", f"{a.dir}/akurasi_per_epoch_{model}.png", f"{a.dir}/confusion_{model}.png")


if __name__ == "__main__":
    main()
