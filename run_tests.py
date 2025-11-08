import os
from train_rl import train_and_signal

os.environ["OUTPUT_DIR"] = "out"
os.environ["MODEL_DIR"]  = "models"

sig = train_and_signal()
assert sig["action"] in ("buy","sell","flat")
assert os.path.exists("out/latest.json")
assert os.path.exists("models/ppo_mnq.zip")
print("OK: RL smoke test passed.")
