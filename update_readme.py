# update_readme.py
# Met à jour le README.md avec les infos du dernier run (multi-indicateurs)
import json, os, datetime

def render_block(s):
    lines = []
    lines.append("🤖 **Dernier run du bot multi-indicateurs**\n")
    lines.append(f"- **Horodatage (UTC)** : `{s['ts']}`")
    lines.append(f"- **PnL net (USD)** : `${s['pnl_usd']}`")
    lines.append(f"- **Trades** : {s['trades']}")
    lines.append(f"- **Win rate** : {s['winrate_pct']}%")
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

    s = json.load(open("out/summary.json", "r"))
    block = render_block(s)

    readme = "README.md"
    if os.path.exists(readme):
        txt = open(readme, "r", encoding="utf-8").read()
    else:
        txt = ""

    new = []
    replaced = False
    for line in txt.splitlines():
        if line.strip().startswith("🤖 **Dernier run"):
            replaced = True
            break
        new.append(line)

    if replaced:
        newtxt = "\n".join(new) + "\n" + block + "\n"
    else:
        newtxt = block + "\n\n" + txt

    with open(readme, "w", encoding="utf-8") as f:
        f.write(newtxt)
    print("✅ README.md updated.")

if __name__ == "__main__":
    main()
