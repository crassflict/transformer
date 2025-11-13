name: Build Dataset

on:
  workflow_dispatch:   # bouton "Run workflow" manuel

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run dataset builder
        run: |
          python ml_build_dataset.py

      - name: Upload dataset as artifact
        uses: actions/upload-artifact@v3
        with:
          name: dataset-output
          path: data/
