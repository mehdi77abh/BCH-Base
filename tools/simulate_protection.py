from collections import defaultdict
import random
import torch
from tools.ieee754 import bitFLIP_v3_tensor

def generate_error_positions(total_bits, ber, seed=None, ignore_sign_bit=True):
    """Same as before – generates global bit positions."""
    if seed is not None:
        random.seed(seed)

    num_errors = int(round(total_bits * ber))

    if ignore_sign_bit:
        total_weights = total_bits // 32
        valid_positions_count = total_weights * 31
        indices = random.sample(range(valid_positions_count), num_errors)
        positions = set()
        for idx in indices:
            weight_idx = idx // 31
            bit_offset = idx % 31
            global_pos = weight_idx * 32 + (bit_offset + 1)
            positions.add(global_pos)
        return positions
    else:
        if num_errors >= total_bits:
            return set(range(total_bits))
        return set(random.sample(range(total_bits), num_errors))

def positions_to_weight_bit_maps(error_set, bits_per_weight=32):
    """Convert global bit positions to dict {weight_idx: set of bit positions}."""
    wmap = defaultdict(set)
    for pos in error_set:
        weight_idx = pos // bits_per_weight
        bit_pos = pos % bits_per_weight
        wmap[weight_idx].add(bit_pos)
    return wmap

def apply_flips(flat_weights_tensor, flips_per_weight):
    """Apply flips to flat_weights_tensor in batches."""
    if not flips_per_weight:
        return flat_weights_tensor

    weight_indices = list(flips_per_weight.keys())
    values_to_flip = flat_weights_tensor[weight_indices]
    positions_list = [list(flips_per_weight[idx]) for idx in weight_indices]

    flipped_values = bitFLIP_v3_tensor(values_to_flip, positions_list)
    flat_weights_tensor[weight_indices] = flipped_values
    return flat_weights_tensor

def simulate_tmr(protection, error_maps):
    """
    TMR: flip bits that appear in at least 2 of the 3 error sets,
    but only for TMR‑protected weights.
    protection: 1D int tensor (0=None,1=TMR,2=BCH)
    error_maps: list of 3 sets of global bit positions
    """
    bits_per_weight = 32
    # Find all weight indices that appear in any error map
    affected_weights = set()
    for err in error_maps:
        for pos in err:
            affected_weights.add(pos // bits_per_weight)

    # Intersect with TMR‑protected weights
    tmr_indices = set(torch.where(protection == 1)[0].tolist())
    relevant = affected_weights & tmr_indices
    if not relevant:
        return defaultdict(set)

    # Convert error sets to weight→bits dicts
    maps = [positions_to_weight_bit_maps(err, bits_per_weight) for err in error_maps]

    flips = defaultdict(set)
    for w_idx in relevant:
        # Collect all bits that appear for this weight in any map
        all_bits = set()
        for m in maps:
            all_bits.update(m.get(w_idx, set()))
        # Count occurrences per bit
        for bit in all_bits:
            count = sum(1 for m in maps if bit in m.get(w_idx, set()))
            if count >= 2:
                flips[w_idx].add(bit)
    return flips

def simulate_bch(protection, error_map1):
    """
    BCH: if a BCH‑protected weight has >2 erroneous bits (from E1), flip all of them.
    Otherwise, no flips (error corrected).
    """
    bits_per_weight = 32
    affected_weights = set(pos // bits_per_weight for pos in error_map1)
    bch_indices = set(torch.where(protection == 2)[0].tolist())
    relevant = affected_weights & bch_indices
    if not relevant:
        return defaultdict(set)

    err_map = positions_to_weight_bit_maps(error_map1, bits_per_weight)
    flips = defaultdict(set)
    for w_idx in relevant:
        bits = err_map.get(w_idx, set())
        if len(bits) > 2:
            flips[w_idx] = bits
    return flips

def simulate_unprotected(protection, error_map1):
    """
    Unprotected: all erroneous bits from E1 affecting unprotected weights are flipped.
    """
    bits_per_weight = 32
    affected_weights = set(pos // bits_per_weight for pos in error_map1)
    none_indices = set(torch.where(protection == 0)[0].tolist())
    relevant = affected_weights & none_indices
    if not relevant:
        return defaultdict(set)

    err_map = positions_to_weight_bit_maps(error_map1, bits_per_weight)
    flips = defaultdict(set)
    for w_idx in relevant:
        bits = err_map.get(w_idx, set())
        if bits:
            flips[w_idx] = bits
    return flips