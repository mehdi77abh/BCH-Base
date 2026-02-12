import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./VGG11.pt", map_location=device, weights_only=False)
model.eval()

for name, param in model.named_parameters():
    # and param.dim() == 4
    if 'weight' in name and 'norm' not in name:
        layer_name = name.replace('.weight', '')
        param_flat = param.view(-1)
        out_channels = param.shape[0]
        # Number of weight elements per filter
        print(f"OUT Channels :{out_channels}")

        elems_per_filter = param[0].numel()
        print(f"Elems Per Filter : {elems_per_filter}")

        # for fidx in range(out_channels):
        #     start = fidx * elems_per_filter
        #     end = start + elems_per_filter
        #     print(f"START : {start} END:{end}")
        #     protection = protection_map.get((layer_name, fidx), 'None')
        #     for i in range(start, end):
        #         flat_weights_list.append(param_flat[i].item())
        #         weights_info.append({
        #             'global_idx': global_idx,
        #             'param_ref': param,
        #             'param_flat_idx': i,
        #             'protection': protection
        #         })
        #         global_idx += 1
# # Save

