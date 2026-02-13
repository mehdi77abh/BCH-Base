from collections import defaultdict
import random
import torch
from tools.ieee754 import bitFLIP_v3_tensor   # unchanged from your code

def generate_error_positions(total_bits, ber, seed=None):
    # total_bits must be total_weights * 31
    if seed is not None:
        random.seed(seed)
    num_errors = int(total_bits * ber)          # truncation, as in OLD
    indices = random.sample(range(total_bits), num_errors)
    return set(indices)                          # raw positions (0 … total_bits-1)


def positions_to_weight_bits(error_set):
    """Convert global bit positions to dict {weight_idx: set of mutable bits (0..30)}."""
    wmap = defaultdict(set)
    for pos in error_set:
        weight_idx = pos // 31
        bit = pos % 31          # 0‑based mutable bit
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



def simulate_old_style(protected_indices, E1, E2, E3, flat_weights):
    # Convert error sets to weight→bits maps
    maps = [positions_to_weight_bits(err) for err in (E1, E2, E3)]

    # Count occurrences of (weight, bit) across all three sets
    counter = defaultdict(int)
    for m in maps:
        for w, bits in m.items():
            for b in bits:
                counter[(w, b)] += 1

    # TMR flips: count ≥2 and weight in protected
    tmr_flips = defaultdict(set)
    for (w, b), cnt in counter.items():
        if cnt >= 2 and w in protected_indices:
            tmr_flips[w].add(b)

    # Single flips: all bits from E1 whose weight is NOT protected
    single_flips = defaultdict(set)
    e1_map = maps[0]   # positions from E1
    for w, bits in e1_map.items():
        if w not in protected_indices:
            single_flips[w].update(bits)

    # Merge flips (they are disjoint because of the protected condition)
    all_flips = defaultdict(set)
    for d in (tmr_flips, single_flips):
        for w, bits in d.items():
            all_flips[w].update(bits)

    # Remove flips for zero‑valued weights (OLD behaviour)
    zero_weights = [w for w in all_flips if flat_weights[w] == 0]
    for w in zero_weights:
        del all_flips[w]

    return all_flips