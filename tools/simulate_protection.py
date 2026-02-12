from collections import defaultdict
import random

from tools.ieee754 import bitFLIP_v3_tensor


# ----------------------------------------------------------------------
#  Error generation and protection simulation
# ----------------------------------------------------------------------
def generate_error_positions(total_bits, ber, seed=None, ignore_sign_bit=True):
    """
    Generate a set of distinct global bit indices (0 .. total_bits-1)
    where a bit flip occurs. Number of errors = round(total_bits * ber).
    Uses random.sample (without replacement).
    """
    if seed is not None:
        random.seed(seed)


    num_errors = int(round(total_bits * ber))

    if ignore_sign_bit:
        total_weights = total_bits // 32          # total_bits is multiple of 32
        valid_positions_count = total_weights * 31
        # Sample from the reduced index space
        indices = random.sample(range(valid_positions_count), num_errors)
        positions = set()
        for idx in indices:
            weight_idx = idx // 31
            bit_offset = idx % 31
            # Skip the sign bit (bit 0) -> add 1
            global_pos = weight_idx * 32 + (bit_offset + 1)
            positions.add(global_pos)
        return positions
    else:
        # Original behaviour – all bits are eligible
        if num_errors >= total_bits:
            return set(range(total_bits))
        return set(random.sample(range(total_bits), num_errors))


def apply_flips(flat_weights_tensor, flips_per_weight):
    """
    Apply the accumulated bit flips to the flat_weights tensor.
    flips_per_weight: dict {weight_global_idx: set of bit positions}
    Uses bitFLIP_v3_tensor on batches of weights.
    """
    if not flips_per_weight:
        return flat_weights_tensor

    # Group weights by the number of flips (for batching)
    weight_indices = list(flips_per_weight.keys())
    # Extract the values to be flipped
    values_to_flip = flat_weights_tensor[weight_indices]
    # Build list of position lists
    positions_list = [list(flips_per_weight[idx]) for idx in weight_indices]

    # Apply bit flips
    flipped_values = bitFLIP_v3_tensor(values_to_flip, positions_list)

    # Write back
    flat_weights_tensor[weight_indices] = flipped_values
    return flat_weights_tensor


def positions_to_weight_bit_maps(error_set, bits_per_weight=32):
    """
    Convert a set of global bit positions into a dict:
        weight_global_idx -> set of bit positions (0..bits_per_weight-1)
    """
    wmap = defaultdict(set)
    for pos in error_set:
        weight_idx = pos // bits_per_weight
        bit_pos = pos % bits_per_weight
        wmap[weight_idx].add(bit_pos)
    return wmap


def simulate_tmr(weights_info, flat_weights, error_maps):
    """
    TMR protection: a bit is flipped if it appears in at least 2 of the 3 error sets
    AND the weight is TMR protected.
    Returns a dict: weight_global_idx -> set of bit positions to flip.
    """
    bits_per_weight = 32
    # Build a mapping from weight_idx to its protection
    prot = {w['global_idx']: w['protection'] for w in weights_info}

    # Convert each error set to weight->bits dict
    maps = [positions_to_weight_bit_maps(err, bits_per_weight) for err in error_maps]

    flips = defaultdict(set)

    # For each weight that is TMR protected
    for w_idx, protection in prot.items():
        if protection != 'TMR':
            continue
        # Collect all bit positions that appear in any of the three sets for this weight
        all_bits = set()
        for m in maps:
            all_bits.update(m.get(w_idx, set()))
        # For each bit, count in how many sets it appears
        for bit in all_bits:
            count = sum(1 for m in maps if bit in m.get(w_idx, set()))
            if count >= 2:
                flips[w_idx].add(bit)
    return flips

def simulate_bch(weights_info, flat_weights, error_map1):
    """
    BCH(7,4,2) simulation: if a weight has >2 erroneous bits (from E1),
    all those bits are flipped. Otherwise, no flip (error corrected).
    """
    bits_per_weight = 32
    prot = {w['global_idx']: w['protection'] for w in weights_info}
    err_map = positions_to_weight_bit_maps(error_map1, bits_per_weight)
    flips = defaultdict(set)
    for w_idx, protection in prot.items():
        if protection != 'BCH':
            continue
        bits = err_map.get(w_idx, set())
        if len(bits) > 2:
            flips[w_idx] = bits
    return flips

def simulate_unprotected(weights_info, flat_weights, error_map1):
    """
    Unprotected weights: all erroneous bits from E1 are flipped.
    """
    bits_per_weight = 32
    prot = {w['global_idx']: w['protection'] for w in weights_info}
    err_map = positions_to_weight_bit_maps(error_map1, bits_per_weight)
    flips = defaultdict(set)
    for w_idx, protection in prot.items():
        if protection == 'None':
            bits = err_map.get(w_idx, set())
            if bits:
                flips[w_idx] = bits
    return flips
