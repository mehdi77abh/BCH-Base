import torch


# ----------------------------------------------------------------------
#  Flattening / unflattening model weights
# ----------------------------------------------------------------------

def flatten_model(model, protection_map):
    """
    Flatten all convolutional weight parameters into a 1D tensor of float32.
    Also build weights_info list for later unflattening and protection lookup.

    Returns:
        flat_weights: torch.Tensor of shape (num_weights,), dtype=torch.float32
        weights_info: list of dict, each with keys:
            'global_idx': index in flat_weights
            'param_ref': reference to the parameter tensor
            'param_flat_idx': index in the flattened parameter (param.view(-1))
            'protection': 'TMR'/'BCH'/'None'
    """
    flat_weights_list = []
    weights_info = []
    global_idx = 0

    for name, param in model.named_parameters():
        # and param.dim() == 4
        if 'weight' in name and 'norm' not in name:
            layer_name = name.replace('.weight', '')
            param_flat = param.view(-1)
            out_channels = param.shape[0]
            # Number of weight elements per filter
            elems_per_filter = param[0].numel()
            for fidx in range(out_channels):
                start = fidx * elems_per_filter
                end = start + elems_per_filter
                protection = protection_map.get((layer_name, fidx), 'None')
                for i in range(start, end):
                    flat_weights_list.append(param_flat[i].item())
                    weights_info.append({
                        'global_idx': global_idx,
                        'param_ref': param,
                        'param_flat_idx': i,
                        'protection': protection
                    })
                    global_idx += 1

                    # Save

    flat_weights = torch.tensor(flat_weights_list, dtype=torch.float32)
    return flat_weights, weights_info


def unflatten_model(model, flat_weights_tensor, weights_info):
    """
    Write back the modified flat_weights into the model parameters.
    """
    for w_info in weights_info:
        param = w_info['param_ref']
        flat_idx = w_info['param_flat_idx']
        new_val = flat_weights_tensor[w_info['global_idx']]
        param.view(-1)[flat_idx] = new_val
