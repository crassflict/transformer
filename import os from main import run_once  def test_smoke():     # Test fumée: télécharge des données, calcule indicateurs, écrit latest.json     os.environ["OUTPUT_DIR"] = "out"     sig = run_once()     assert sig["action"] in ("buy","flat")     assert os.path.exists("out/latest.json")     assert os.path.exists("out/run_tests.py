import os
from main import run_once

def test_smoke():
    # Test fumée: télécharge des données, calcule indicateurs, écrit latest.json
    os.environ["OUTPUT_DIR"] = "out"
    sig = run_once()
    assert sig["action"] in ("buy","flat")
    assert os.path.exists("out/latest.json")
    assert os.path.exists("out/report.csv")

if __name__ == "__main__":
    test_smoke()
    print("OK: smoke test passed.")
