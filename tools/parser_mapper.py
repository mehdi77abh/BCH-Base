import json
import re

def parse_sensitivity_key_flat(key):
    """For flat format: 'filter0,layer1(features.0)' -> ('features.0', 0)"""
    layer_part = key.split('(')[1].split(')')[0]
    filter_part = key.split(',')[0]
    filter_idx = int(filter_part.replace('filter', ''))
    return layer_part, filter_idx

def load_sensitivity(rank_path, model_name=None, layer_mapping=None):
    """
    Load sensitivity JSON and return dict {(actual_layer, filter_idx): score}.
    - If nested (first scheme), layer_mapping must map descriptive keys to actual names.
    - If flat (second scheme), no mapping needed.
    ('features.0', 0): 177.47
    ('features.0', 1): 1.0
    """
    with open(rank_path, 'r') as f:
        data = json.load(f)

    sensitivity = {}
    first_key = next(iter(data))

    # Detect format
    if isinstance(data[first_key], dict):
        # Nested format: { "Layer_1_Conv2d_3to64": {"Filter_0": 177.47, ...}, ... }
        if layer_mapping is None:
            raise ValueError("Nested JSON format requires layer_mapping.")
        for layer_key, filters_dict in data.items():
            actual_layer = layer_mapping.get(layer_key, layer_key)
            for filter_key, score in filters_dict.items():
                filter_idx = int(filter_key.replace('Filter_', ''))
                sensitivity[(actual_layer, filter_idx)] = float(score)
    else:
        # Flat format: { "filter0,layer1(features.0)": 2.335, ... }
        for key, score in data.items():
            layer, fidx = parse_sensitivity_key_flat(key)
            sensitivity[(layer, fidx)] = float(score)

    return sensitivity


