# plot_results.py — génère un graphique du PnL net historique
# Lit les fichiers out/run_*.txt, extrait "PnL net (USD): ...", trace et écrit out/performance.png

import os, re, glob, datetime as dt

# Matplotlib en mode headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "out"
IMG_PATH = os.path.join(OUT_DIR, "performance.png")
RUN_GLOB = os.path.join(OUT_DIR, "run_*.txt")

def parse_runs():
    runs = []
    for path in sorted(glob.glob(RUN_GLOB)):
        # timestamp dans le nom: run_YYYYMMDDTHHMMSSZ.txt
        base = os.path.basename(path)
        m_ts = re.match(r"run_(\d{8}T\d{6}Z)\.txt", base)
        ts = None
        if m_ts:
            try:
                ts = dt.datetime.strptime(m_ts.group(1), "%Y%m%dT%H%M%SZ")
            except Exception:
                ts = None

        pnl = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "PnL net (USD):" in line:
                    m = re.search(r"PnL net \(USD\):\s*([+-]?\d+(?:\.\d+)?)", line)
                    if m:
                        pnl = float(m.group(1))
                        break
        if ts is not None and pnl is not None:
            runs.append((ts, pnl, path))

    runs.sort(key=lambda x: x[0])
    return runs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = parse_runs()
    if not runs:
        print("No runs found to plot. Skipping.")
        return

    xs = [r[0] for r in runs]
    ys = [r[1] for r in runs]

    # Création du graphique
    plt.figure(figsize=(8,4.5), dpi=150)
    plt.plot(xs, ys, marker="o", linewidth=1)
    plt.title("Bot PnL net (USD) au fil des runs")
    plt.xlabel("Run (horodatage UTC)")
    plt.ylabel("PnL net (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_PATH)
    print(f"Wrote {IMG_PATH} with {len(ys)} points")

if __name__ == "__main__":
    main()
