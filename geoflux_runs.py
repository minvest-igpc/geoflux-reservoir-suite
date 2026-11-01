"""GeoFlux run records: every calculation can be saved with its inputs, method,
engine version and random seed, so it can be audited and reproduced."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ENGINE_VERSION = "0.3.0"
SCHEMA_VERSION = 1


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records", date_format="iso"))
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def hash_inputs(inputs):
    """SHA-256 of the inputs in a canonical form (key order does not matter)."""
    s = json.dumps(inputs, sort_keys=True, default=_default, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_record(method, inputs, outputs, seed=None):
    return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": method, "seed": seed, "inputs_sha256": hash_inputs(inputs),
            "inputs": inputs, "outputs": outputs}


def verify_record(rec):
    """True if the stored inputs still match the stored hash."""
    return hash_inputs(rec["inputs"]) == rec["inputs_sha256"]


def record_json(rec):
    return json.dumps(rec, default=_default, indent=2)


# END OF FILE: geoflux_runs.py