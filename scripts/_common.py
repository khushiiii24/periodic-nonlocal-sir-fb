import json, os, sys, time, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, np.integer):
        return int(o)
    return o

def save(name, obj):
    path = os.path.join(RES, name + ".json")
    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=1)
    print("saved", path)

def load(name):
    with open(os.path.join(RES, name + ".json")) as fh:
        return json.load(fh)
