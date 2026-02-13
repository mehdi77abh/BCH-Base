import math
from tools.parser_mapper import load_sensitivity

def get_all_filters(model):
    """Return list of (layer_name, filter_idx) for all conv filters."""
    filters = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name and param.dim() == 4:
            layer_name = name.replace('.weight', '')
            out_channels = param.shape[0]
            for fidx in range(out_channels):
                filters.append((layer_name, fidx))
    return filters

def compute_protection(model, sensitivity, tmr_percent, bch_percent):
    """
    Returns a dict {(layer, filter_idx): 'TMR'|'BCH'|'None'}.
    """
    all_filters = get_all_filters(model)
    # Assign scores (default 0 if filter not in sensitivity)
    scored = [(f, sensitivity.get(f, 0.0)) for f in all_filters]
    scored.sort(key=lambda x: x[1], reverse=True)
    sorted_filters = [f for f, _ in scored]

    n = len(sorted_filters)
    tmr_cnt = int(math.ceil(n * tmr_percent / 100.0))
    bch_cnt = int(math.ceil(n * bch_percent / 100.0))

    protection = {}
    for i, f in enumerate(sorted_filters):
        if i < tmr_cnt:
            protection[f] = 'TMR'
        elif i < tmr_cnt + bch_cnt:
            protection[f] = 'BCH'
        else:
            protection[f] = 'None'
    return protection