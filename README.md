# Trading Bot (signals -> latest.json)

Pipeline simple et stable :
- Récupère OHLC Kraken (public API)
- Calcule EMA21/EMA55 + RSI14
- Génère un signal `buy/flat`
- Écrit `out/latest.json` + `out/report.csv`

## Utilisation locale
```bash
pip install -r requirements.txt
python main.py
