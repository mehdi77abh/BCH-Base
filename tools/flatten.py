import torch

def flatten_model(model, protection_map):
    """
    Flatten conv weights and build a protection tensor (int codes).
    Returns:
        flat_weights: 1D torch.Tensor of all conv weights (float32)
        protection:   1D torch.Tensor of same length: 0=None, 1=TMR, 2=BCH
        param_info:   list of dicts for unflattening
    """
    code = {'None': 0, 'TMR': 1, 'BCH': 2}
    flat_pieces = []
    prot_pieces = []
    param_info = []
    global_start = 0

    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            layer_name = name.replace('.weight', '')
            out_channels = param.shape[0]
            wpf = param[0].numel()                     # weights per filter
            total_weights = param.numel()

            # Build per‑filter protection codes
            filter_codes = []
            for fidx in range(out_channels):
                prot = protection_map.get((layer_name, fidx), 'None')
                filter_codes.append(code[prot])

            # Repeat each code for the number of elements in that filter
            prot_tensor = torch.tensor(filter_codes, dtype=torch.int)
            prot_tensor = prot_tensor.repeat_interleave(wpf)

            flat_pieces.append(param.view(-1))
            prot_pieces.append(prot_tensor)
            param_info.append({
                'param_ref': param,
                'start': global_start,
                'end': global_start + total_weights,
                'shape': param.shape
            })
            global_start += total_weights

    flat_weights = torch.cat(flat_pieces)
    protection = torch.cat(prot_pieces)
    return flat_weights, protection, param_info

def unflatten_model(model, flat_weights, param_info):
    for info in param_info:
        param = info['param_ref']
        start = info['start']
        end = info['end']
        param.data.copy_(flat_weights[start:end].view(info['shape']))