import torch

def flatten_model(model, protection_map):
    """
    Flatten all convolutional weight parameters into a 1D tensor.
    Also build a protection tensor (int codes) and param_info for unflattening.

    Args:
        model: PyTorch model
        protection_map: dict {(layer_name, filter_idx): 'TMR'|'BCH'|'None'}

    Returns:
        flat_weights: 1D torch.Tensor of all weights (float32)
        protection:   1D torch.Tensor of same length with codes 0=None,1=TMR,2=BCH
        param_info:   list of dicts with keys: 'param_ref', 'start', 'end', 'shape'
    """
    flat_weights_list = []
    protection_list = []
    param_info = []
    global_start = 0

    code_map = {'None': 0, 'TMR': 1, 'BCH': 2}

    for name, param in model.named_parameters():
        # Only conv weights (4D) and exclude norm layers
        if 'weight' in name and 'norm' not in name and param.dim() == 4:
            layer_name = name.replace('.weight', '')
            out_channels = param.shape[0]
            param_flat = param.view(-1)
            num_weights = param_flat.numel()
            elems_per_filter = num_weights // out_channels

            # Build per‑filter protection codes
            filter_codes = []
            for fidx in range(out_channels):
                prot = protection_map.get((layer_name, fidx), 'None')
                filter_codes.append(code_map[prot])

            # Repeat each code for the number of elements in that filter
            filter_codes_tensor = torch.tensor(filter_codes, dtype=torch.int)
            prot_tensor = filter_codes_tensor.repeat_interleave(elems_per_filter)

            flat_weights_list.append(param_flat)
            protection_list.append(prot_tensor)
            param_info.append({
                'param_ref': param,
                'start': global_start,
                'end': global_start + num_weights,
                'shape': param.shape
            })
            global_start += num_weights

    flat_weights = torch.cat(flat_weights_list)
    protection = torch.cat(protection_list)
    return flat_weights, protection, param_info


def unflatten_model(model, flat_weights, param_info):
    """
    Write back the modified flat_weights into the model parameters using param_info.
    """
    for info in param_info:
        param = info['param_ref']
        start = info['start']
        end = info['end']
        param.data.copy_(flat_weights[start:end].view(info['shape']))