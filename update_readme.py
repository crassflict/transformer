# update_readme.py — met à jour README.md avec le dernier résumé du bot
import json, os, datetime

SUMMARY_PATH = "out/summary.json"
README_PATH = "README.md"

START = "<!-- BOT-SUMMARY:START -->"
END   = "<!-- BOT-SUMMARY:END -->"

def load_summary():
    if not os.path.isfile(SUMMARY_PATH):
        return None
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def render_block(s):
    lines = []
    lines.append(START)
    lines.append("")
    lines.append("## 🤖 Dernier run du bot")
    lines.append("")
    lines.append(f"- **Horodatage (UTC)**: `{s['ts']}`")
    lines.append(f"- **PnL net**: **${s['pnl_usd']:.2f}**")
    lines.append(f"- **Trades**: **{s['trades']}**")
    lines.append(f"- **Win rate**: **{s['winrate_pct']:.2f}%**")
    lines.append("")
    lines.append("| Paramètre | Valeur |")
    lines.append("|---|---:|")
    lines.append(f"| EMA rapide | {s['ema_fast']} |")
    lines.append(f"| EMA lente | {s['ema_slow']} |")
    lines.append(f"| RSI buy | {s['rsi_buy']:.1f} |")
    lines.append(f"| RSI sell | {s['rsi_sell']:.1f} |")
    lines.append("")
    lines.append(f"_Source des données_: `{s['data_source']}`")
    lines.append("")
    lines.append(END)
    lines.append("")
    return "\n".join(lines)

def insert_or_replace_block(readme, block):
    if START in readme and END in readme:
        head = readme.split(START)[0]
        tail = readme.split(END)[1]
        return head + block + tail
    else:
        # Ajoute en haut si pas de bloc
        return block + "\n" + readme

def main():
    s = load_summary()
    if not s:
        print("No summary.json yet; skip README update.")
        return
    if os.path.isfile(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme = f.read()
    else:
        readme = "# transformer\n\n"
    block = render_block(s)
    new_readme = insert_or_replace_block(readme, block)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_readme)
    print("README.md updated.")

if __name__ == "__main__":
    main()
