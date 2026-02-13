import math
import json


# ----------------------------------------------------------------------
#  Helper functions for parsing and mapping
# ----------------------------------------------------------------------
def parse_sensitivity_key(key):
    """
    Key format: "filter<filter_idx>,layer<layer_name>(<actual_layer_name>)"
    Example: "filter0,layer1(features.0)" -> ('features.0', 0)
    """
    # Extract part after '(' and before ')'
    layer_part = key.split('(')[1].split(')')[0]  # features.0
    filter_part = key.split(',')[0]  # filter0
    filter_idx = int(filter_part.replace('filter', ''))
    return layer_part, filter_idx

def load_tmr_sensitivity(rank_dir, model_name, layer_mapping=None):
    """
    Load sensitivity JSON and return dict {(layer_name, filter_idx): score}.
    Handles both nested (first) and flat (second) formats.
    If layer_mapping is provided, maps JSON layer keys to actual layer names.
    """
    layer_mapping = {
        "Layer_1_Conv2d_3to64": "features.0",
        "Layer_2_Conv2d_64to128": "features.3",
        "Layer_3_Conv2d_128to256": "features.6",
        "Layer_4_Conv2d_256to256": "features.8",
        "Layer_5_Conv2d_256to512": "features.11",
        "Layer_6_Conv2d_512to512": "features.13",
        "Layer_7_Conv2d_512to512": "features.15",
        "Layer_8_Conv2d_512to512": "features.17",
    }
    with open(rank_dir, 'r') as f:
        data = json.load(f)

    sensitivity = {}
    first_key = next(iter(data))
    if isinstance(data[first_key], dict):
        # Nested format (first scheme)
        for layer_key, filters_dict in data.items():
            actual_layer = layer_mapping.get(layer_key, layer_key) if layer_mapping else layer_key
            for filter_key, score in filters_dict.items():
                filter_idx = int(filter_key.replace('Filter_', ''))
                sensitivity[(actual_layer, filter_idx)] = float(score)
    else:
        # Flat format (second scheme)
        for key, score in data.items():
            layer, fidx = parse_sensitivity_key(key)
            sensitivity[(layer, fidx)] = float(score)
    return sensitivity

def load_sensitivity(rank_dir, model_name):
    """
    Load the sensitivity JSON and return a dict: {(layer_name, filter_idx): score}
    Example : ('features.0', 0): 2.3358620275456214
    """
    path = f"{rank_dir}/{model_name}_sensitivity.json"
    # TODO Change dir to work dynamically
    with open("./final_importance_sorted.json", 'r') as f:
        data = json.load(f)
    sensitivity = {}
    for key, score in data.items():
        layer, fidx = parse_sensitivity_key(key)
        sensitivity[(layer, fidx)] = score

    return sensitivity


def extract_all_filters_from_model(model):
    """
    For convolutional layers (4D weight tensors) extract all (layer_name, filter_idx).
    Only parameters with 'weight' in name and not 'norm' are considered.
    """
    filters = []
    for name, param in model.named_parameters():
        #and param.dim() == 4
        if 'weight' in name and 'norm' not in name :
            # Name may be like 'features.0.weight' -> layer name 'features.0'
            # Remove trailing '.weight'
            layer_name = name.replace('.weight', '')
            out_channels = param.shape[0]
            for fidx in range(out_channels):
                filters.append((layer_name, fidx))
    return filters


def compute_filter_protection(model, sensitivity_map, tmr_percent, bch_percent):
    """
    Assign protection type to every filter based on sensitivity ranking.
    Returns dict: {(layer, filter_idx): 'TMR'|'BCH'|'None'}
    """
    all_filters = extract_all_filters_from_model(model)
    # Score for each filter: use sensitivity_map if exists, else 0
    scored = [(f, sensitivity_map.get(f, 0.0)) for f in all_filters]
    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)
    sorted_filters = [f for f, _ in scored]
    n = len(sorted_filters)
    tmr_cnt = math.ceil(n * tmr_percent / 100)
    bch_cnt = math.ceil(n * bch_percent / 100)
    protection = {}
    for i, f in enumerate(sorted_filters):
        if i < tmr_cnt:
            protection[f] = 'TMR'
        elif i < tmr_cnt + bch_cnt:
            protection[f] = 'None'
        else:
            protection[f] = 'None'
    return protection
