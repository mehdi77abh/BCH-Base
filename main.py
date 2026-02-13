import json
import torch
from collections import defaultdict

from tqdm import tqdm

from test import save_data_to_json, sort_json_values
from tools.evaluation import evaluate
from tools.flatten import flatten_model, unflatten_model
from tools.log_to_csv import log_to_csv
from tools.parser_mapper import load_sensitivity
from tools.protection import compute_protection, load_protected_indices_from_json
from tools.simulate_protection import generate_error_positions, simulate_tmr, simulate_bch, simulate_unprotected, \
    apply_flips, simulate_old_style


def load_model(model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location=device)
    model.eval()
    return model


def run_experiment_repetitions(config, num_repetitions=15):
    # print(f"\n=== {config['model_name']} | BER={config['ber']:.0e} | "f"TMR={config['tmr_per']}% BCH={config['bch_high_per']}% | {num_repetitions} reps ===")

    # Load model
    model = load_model(config['model_dir'])

    # For first JSON scheme, you need a layer mapping.
    # Example for VGG11 on CIFAR-100:
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
    # Load sensitivity (auto‑detects format)
    sensitivity = load_sensitivity(config['rank_dir'], layer_mapping=layer_mapping)

    # Compute which filters get TMR, BCH, None
    protection_map = compute_protection(model, sensitivity,
                                        config['tmr_per'],
                                        config['bch_high_per'])

    # Flatten model and build protection tensor
    flat_weights_clean, protection, param_info = flatten_model(model, protection_map)
    # Total number of convolution weights (including all weight tensors now)
    total_weights = len(flat_weights_clean)
    total_bits = total_weights * 31  # OLD uses 31 bits per weight

    # Load protected indices from the JSON file (same path as OLD)
    json_path = "./filter_indices_35pct.json"
    protected_indices = load_protected_indices_from_json(json_path, total_weights)

    # Keep original state dict for resetting
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    csv_path = f"{config['model_name']}_results.csv"

    for rep in tqdm(range(num_repetitions)):
        flat_weights = flat_weights_clean.clone()

        seed_base = rep * 100
        E1 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 42)
        E2 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 43)
        E3 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 44)

        # Get flips from each protection scheme
        flips = simulate_old_style(protected_indices, E1, E2, E3, flat_weights_clean)

        # Apply flips
        flat_weights = apply_flips(flat_weights, flips)

        # Restore original model and write back
        model.load_state_dict(original_state_dict)
        unflatten_model(model, flat_weights, param_info)

        # Evaluate
        acc, tacc, prec, rec, conf, sub_conf, acc_50 = evaluate(
            model, config['dataset_name'], config['dataset_dir']
        )

        # Log
        log_to_csv(csv_path,
                   config['ber'],
                   config['tmr_per'],
                   rep,
                   acc, tacc, prec, rec, conf, sub_conf, acc_50)

        print(f"   Rep {rep + 1:2d}: Acc = {acc:.4f}")

    print(f"=== Finished ===")


if __name__ == '__main__':
    with open('configs.json') as f:
        configs = json.load(f)
    for config in configs:
        print(f"MODE:{config['model_dir'],} BER : {config['ber']} TMR Perc : {config['tmr_per']}")
        run_experiment_repetitions(config, 1)
