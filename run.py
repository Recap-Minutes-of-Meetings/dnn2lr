import subprocess

files = [
    "./data/get_config.py",
    
    "./models/nn/make_config.py",
    "./models/nn/run0.py",
    
    "./models/lgb/make_config.py",
    "./models/lgb/run0.py",
    
    "./models/linear/make_config.py",
    "./models/linear/run0.py",
    
    "./evaluation/get_metrics.py",
    "./evaluation/get_summary.py",
    
]

for file in files:
    subprocess.call(['python3', file], cwd=file)
