import json
import numpy as np

def numpy_to_native(data):
    """Transforms NumPy types to Python native for JSON serialization"""
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_native(item) for item in data]
    return data

def safe_json_dumps(data):
    """Serialize data"""
    return json.dumps(numpy_to_native(data))