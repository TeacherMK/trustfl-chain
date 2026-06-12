import subprocess
import sys


cmd = [
    sys.executable,
    "src/main.py",
    "--dataset",
    "fashion_mnist",
    "--rounds",
    "5",
    "--num_clients",
    "20",
    "--clients_per_round",
    "10",
    "--malicious_ratio",
    "0.2",
    "--attack",
    "label_flip",
    "--defense",
    "trustfl_chain",
    "--non_iid",
    "true",
]
raise SystemExit(subprocess.call(cmd))
