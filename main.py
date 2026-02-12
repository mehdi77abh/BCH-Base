import json
from collections import defaultdict
import torch

from tools.evaluation import evaluate
from tools.flatten import flatten_model, unflatten_model
from tools.log_to_csv import log_to_csv
from tools.parser_mapper import load_sensitivity, compute_filter_protection, load_tmr_sensitivity
from tools.simulate_protection import simulate_tmr, simulate_bch, simulate_unprotected, generate_error_positions, \
    apply_flips




# ----------------------------------------------------------------------
#  Main experiment loop
# ----------------------------------------------------------------------
# def run_experiment(config,num_repetitions=15):
#     print(f"\n=== Running experiment for {config['model_name']} ===")
#
#     # 1. Load model and sensitivity
#     model = load_model(config['model_name'], config['model_dir'])
#     sensitivity = load_sensitivity(config['rank_dir'], config['model_name'])
#
#     # 2. Assign protection types
#     protection_map = compute_filter_protection(
#         model,
#         sensitivity,
#         config['tmr_per'],
#         config['bch_high_per']
#     )
#
#     # 3. Flatten
#     flat_weights, weights_info = flatten_model(model, protection_map)
#     total_weights = len(flat_weights)
#     total_bits = total_weights * 32
#
#     original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
#     csv_path = f"{config['model_name']}_results.csv"
#     print(f"Total conv weights: {total_weights}, bits: {total_bits}")
#
#     # 4. Generate three error sets
#
#
#     ber = config['ber']
#     E1 = generate_error_positions(total_bits, ber, seed=42)
#     E2 = generate_error_positions(total_bits, ber, seed=43)
#     E3 = generate_error_positions(total_bits, ber, seed=44)
#
#     # 5. TMR simulation
#     tmr_flips = simulate_tmr(weights_info, flat_weights, [E1, E2, E3])
#
#     # 6. BCH simulation (only if percent > 0)
#     bch_flips = defaultdict(set)
#     if config['bch_high_per'] > 0:
#         bch_flips = simulate_bch(weights_info, flat_weights, E1)
#
#     # 7. Unprotected simulation
#     unprotected_flips = simulate_unprotected(weights_info, flat_weights, E1)
#
#     # 8. Merge flips
#     all_flips = defaultdict(set)
#     for d in [tmr_flips, bch_flips, unprotected_flips]:
#         for k, v in d.items():
#             all_flips[k].update(v)
#
#     # 9. Apply flips
#     flat_weights = apply_flips(flat_weights, all_flips)
#
#     # 10. Write back to model
#     unflatten_model(model, flat_weights, weights_info)
#
#     # 11. Evaluate
#     acc, tacc, prec, rec, conf, sub_conf, acc_50 = evaluate(
#         model, config['dataset_name'], config['dataset_dir']
#     )
#     print(f"Accuracy: {acc:.4f}, Top-5: {tacc:.4f}")
#
#     # 12. Log to per‑model CSV
#     csv_path = f"{config['model_name']}_results.csv"
#     log_to_csv(csv_path, ber, config['tmr_per'], 0,
#                acc, tacc, prec, rec, conf, sub_conf, acc_50)
#
#     return {
#         'model': config['model_name'],
#         'ber': ber,
#         'tmr_per': config['tmr_per'],
#         'accuracy': acc,
#         'top5': tacc
#     }

def run_experiment_repetitions(config, num_repetitions=15):
    print(f"\n=== {config['model_name']} | BER={config['ber']:.0e} | TMR={config['tmr_per']}% | {num_repetitions} reps ===")

    model = load_model(config['model_name'], config['model_dir'])
    sensitivity = load_tmr_sensitivity(config['rank_dir'], config['model_name'])

    protection_map = compute_filter_protection(model, sensitivity,
                                               config['tmr_per'],
                                               config['bch_high_per'])
    print(sensitivity)
    print("="*30)
    print(protection_map)
    print("="*30)
    print(f"\n=== Flatten Start ===")
    # Flatten once, keep clean weights
    flat_weights_clean, weights_info = flatten_model(model, protection_map)
    total_bits = len(flat_weights_clean) * 32
    print(f"\n=== Flatten Finish ===")

    # We will write back to the model each repetition, so keep a reference.
    # But we must restore model to original weights before each write-back.
    # Better: keep the original state_dict.
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # ---- CSV file for this model (created once) ----
    csv_path = f"{config['model_name']}_results.csv"

    # ---- Repetition loop ----
    for rep in range(num_repetitions):
        # 1. Fresh copy of weights
        flat_weights = flat_weights_clean.clone()

        # 2. New error sets with different seeds
        seed_base = rep * 100  # ensures distinct seeds
        E1 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 42)
        E2 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 43)
        E3 = generate_error_positions(total_bits, config['ber'], seed=seed_base + 44)

        # 3. Simulate protections (same functions, use flat_weights)
        tmr_flips = simulate_tmr(weights_info, flat_weights, [E1, E2, E3])
        bch_flips = defaultdict(set)
        if config['bch_high_per'] > 0:
            bch_flips = simulate_bch(weights_info, flat_weights, E1)
        unprotected_flips = simulate_unprotected(weights_info, flat_weights, E1)

        # 4. Merge flips
        all_flips = defaultdict(set)
        for d in [tmr_flips, bch_flips, unprotected_flips]:
            for k, v in d.items():
                all_flips[k].update(v)

        # 5. Apply flips to this flat_weights copy
        flat_weights = apply_flips(flat_weights, all_flips)

        # 6. Restore model to original weights
        model.load_state_dict(original_state_dict)

        # 7. Write back the modified flat_weights into the model
        unflatten_model(model, flat_weights, weights_info)

        # 8. Evaluate
        acc, tacc, prec, rec, conf, sub_conf, acc_50 = evaluate(
            model, config['dataset_name'], config['dataset_dir']
        )

        # 9. Log to CSV (one row per repetition)
        log_to_csv(csv_path,
                   config['ber'],
                   config['tmr_per'],
                   rep,              # iteration number
                   acc, tacc, prec, rec, conf, sub_conf, acc_50)

        print(f"   Rep {rep+1:2d}: Acc = {acc:.4f}")

    print(f"=== Finished {num_repetitions} repetitions ===")

def load_model(model_name, model_dir):
    """
    Dummy model loader – replace with actual loading code.
    Assumes the model architecture is known and weights are stored as .pth.
    """
    # Example: for VGG11 you would import the model class
    # model = torchvision.models.vgg11()
    # state_dict = torch.load(f"{model_dir}/{model_name}.pth")
    # model.load_state_dict(state_dict)
    # model.eval()
    # return model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location=device)
    model.eval()
    return model

    # raise NotImplementedError("Please implement load_model() for your specific model.")


if __name__ == '__main__':
    config_path = "./configs.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    results = []
    REPETITIONS  = 15
    # for cfg in configs:
    run_experiment_repetitions(configs[0], REPETITIONS)


    # Save summary
    with open('hardening_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAll experiments finished. Results saved to hardening_results.json")
