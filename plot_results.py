# plot_results.py — génère un graphique PnL + Win rate
# Lit out/run_*.txt et extrait:
#   PnL net (USD): <float>
#   Win rate: <float>%
# Crée out/performance.png avec 2 axes Y (PnL & Win rate)

import os, re, glob, datetime as dt

# Matplotlib en mode headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "out"
IMG_PATH = os.path.join(OUT_DIR, "performance.png")
RUN_GLOB = os.path.join(OUT_DIR, "run_*.txt")

RE_PNL = re.compile(r"PnL net \(USD\):\s*([+-]?\d+(?:\.\d+)?)")
RE_WR  = re.compile(r"Win rate:\s*([0-9]+(?:\.[0-9]+)?)%")

def parse_runs():
    rows = []
    for path in sorted(glob.glob(RUN_GLOB)):
        base = os.path.basename(path)
        m_ts = re.match(r"run_(\d{8}T\d{6}Z)\.txt", base)
        if not m_ts:
            continue
        try:
            ts = dt.datetime.strptime(m_ts.group(1), "%Y%m%dT%H%M%SZ")
        except Exception:
            continue

        pnl = None
        wr = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if pnl is None:
                    m = RE_PNL.search(line)
                    if m: pnl = float(m.group(1))
                if wr is None:
                    m = RE_WR.search(line)
                    if m: wr = float(m.group(1))
                if pnl is not None and wr is not None:
                    break

        if pnl is not None and wr is not None:
            rows.append((ts, pnl, wr, path))

    rows.sort(key=lambda x: x[0])
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = parse_runs()
    if not data:
        print("No runs found to plot. Skipping.")
        return

    xs = [r[0] for r in data]
    pnls = [r[1] for r in data]
    wrs  = [r[2] for r in data]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=150)

    # Courbe PnL
    l1 = ax1.plot(xs, pnls, marker="o", linewidth=1.5, label="PnL net (USD)")
    ax1.set_xlabel("Run (horodatage UTC)")
    ax1.set_ylabel("PnL net (USD)")
    ax1.grid(True, alpha=0.3)

    # Axe secondaire pour Win rate
    ax2 = ax1.twinx()
    l2 = ax2.plot(xs, wrs, marker="s", linestyle="--", linewidth=1.2, label="Win rate (%)")
    ax2.set_ylabel("Win rate (%)")

    # Légende combinée
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Bot — PnL net (USD) & Win rate (%)")
    fig.tight_layout()
    plt.savefig(IMG_PATH)
    print(f"Wrote {IMG_PATH} with {len(xs)} points")

if __name__ == "__main__":
    main()
