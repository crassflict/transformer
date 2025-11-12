# update_readme.py — met à jour README.md avec le dernier résumé + le graphique PnL
import json, os

IMG_PATH = "out/performance.png"

def render_block(s, has_img):
    lines = []
    lines.append("🤖 **Dernier run du bot multi-indicateurs**\n")
    lines.append(f"- **Horodatage (UTC)** : `{s['ts']}`")
    lines.append(f"- **PnL net (USD)** : `${s['pnl_usd']}`")
    lines.append(f"- **Trades** : {s['trades']}")
    lines.append(f"- **Win rate** : {s['winrate_pct']}%")
    lines.append("")
    if has_img:
        # afficher l’image (equity curve)
        lines.append("![Évolution du PnL](out/performance.png)")
        lines.append("")
    lines.append("| Indicateur | Valeur |")
    lines.append("|:-----------|-------:|")
    lines.append(f"| EMA 9 | {s['ema9']} |")
    lines.append(f"| EMA 21 | {s['ema21']} |")
    lines.append(f"| SMA 50 | {s['sma50']} |")
    lines.append(f"| SMA 200 | {s['sma200']} |")
    lines.append(f"| RSI len | {s['rsi_len']} |")
    lines.append(f"| MACD (fast, slow, signal) | {s['macd']} |")
    lines.append(f"| VWAP window | {s['vwap_win']} |")
    lines.append(f"| Volume MA window | {s['vma_win']} |")
    lines.append(f"| Volume Profile window | {s['vp_win']} |")
    lines.append("")
    lines.append(f"_Source des données :_ `{s['data_source']}`")
    return "\n".join(lines)

def main():
    if not os.path.exists("out/summary.json"):
        print("No summary.json found, skipping README update.")
        return
    s = json.load(open("out/summary.json", "r", encoding="utf-8"))
    has_img = os.path.exists(IMG_PATH)
    block = render_block(s, has_img)

    readme = "README.md"
    txt = open(readme, "r", encoding="utf-8").read() if os.path.exists(readme) else ""

    # Remplace la première section existante "Dernier run du bot multi-indicateurs"
    lines = txt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("🤖 **Dernier run du bot multi-indicateurs**"):
            start = i
            break

    if start is not None:
        # garder ce qui est avant, remplacer la section entière jusqu'à une ligne vide suivie d'un titre ou EOF
        new = lines[:start]
        new.append(block)
        new.append("")  # une ligne vide
        txt = "\n".join(new)
    else:
        txt = block + "\n\n" + txt

    with open(readme, "w", encoding="utf-8") as f:
        f.write(txt)
    print("✅ README.md updated.")

if __name__ == "__main__":
    main()
