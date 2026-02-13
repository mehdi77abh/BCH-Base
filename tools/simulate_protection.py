from collections import defaultdict
import random
import torch
from tools.ieee754 import bitFLIP_v3_tensor   # unchanged from your code

def generate_error_positions(total_bits, ber, seed=None):
    if seed is not None:
        random.seed(seed)

    num_errors = int(round(total_bits * ber))
    total_weights = total_bits // 32
    # Sample from 31 mutable bits per weight
    valid_positions_count = total_weights * 31
    indices = random.sample(range(valid_positions_count), num_errors)
    positions = set()
    for idx in indices:
        weight_idx = idx // 31
        bit_offset = idx % 31
        global_pos = weight_idx * 32 + (bit_offset + 1)   # skip sign bit
        positions.add(global_pos)
    return positions

def positions_to_weight_bits(error_set):
    """Convert global bit positions to dict {weight_idx: set of mutable bits (0..30)}."""
    wmap = defaultdict(set)
    for pos in error_set:
        weight_idx = pos // 32
        bit = pos % 32 - 1          # 0‑based mutable bit
        wmap[weight_idx].add(bit)
    return wmap

def apply_flips(flat_weights, flips_per_weight):
    if not flips_per_weight:
        return flat_weights
    indices = list(flips_per_weight.keys())
    values = flat_weights[indices]
    pos_list = [list(flips_per_weight[i]) for i in indices]
    flipped = bitFLIP_v3_tensor(values, pos_list)
    flat_weights[indices] = flipped
    return flat_weights

def simulate_tmr(protection, E1, E2, E3):
    """
    protection: 1D int tensor (0=None,1=TMR,2=BCH)
    Returns dict {weight_idx: set of mutable bits} for TMR weights with ≥2 errors.
    """
    bits_per_weight = 32
    # Which weight indices appear in any error set?
    affected = set()
    for err in (E1, E2, E3):
        for pos in err:
            affected.add(pos // bits_per_weight)

    # Filter to TMR weights only
    tmr_weights = set(torch.where(protection == 1)[0].tolist())
    relevant = affected & tmr_weights
    if not relevant:
        return defaultdict(set)

    # Convert error sets to weight→bits maps
    maps = [positions_to_weight_bits(err) for err in (E1, E2, E3)]

    flips = defaultdict(set)
    for w in relevant:
        all_bits = set()
        for m in maps:
            all_bits.update(m.get(w, set()))
        for bit in all_bits:
            count = sum(1 for m in maps if bit in m.get(w, set()))
            if count >= 2:
                flips[w].add(bit)
    return flips

def simulate_bch(protection, E1):
    """
    For BCH weights: flip all bits of the weight if it has >2 erroneous bits.
    """
    bits_per_weight = 32
    err_map = positions_to_weight_bits(E1)

    bch_weights = set(torch.where(protection == 2)[0].tolist())
    flips = defaultdict(set)
    for w in bch_weights:
        bits = err_map.get(w, set())
        if len(bits) > 2:
            flips[w] = bits
    return flips

def simulate_unprotected(protection, E1):
    """
    Unprotected weights: flip all bits from E1 that fall on them.
    """
    bits_per_weight = 32
    err_map = positions_to_weight_bits(E1)

    unprotected = set(torch.where(protection == 0)[0].tolist())
    flips = defaultdict(set)
    for w in unprotected:
        bits = err_map.get(w, set())
        if bits:
            flips[w] = bits
    return flips