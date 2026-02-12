import numpy as np
import torch


# ----------------------------------------------------------------------
#  Provided functions – do not modify
# ----------------------------------------------------------------------
def fractional_to_bin(dec_part, length=24):
    mantissa = ''
    for _ in range(length):
        dec_part *= 2
        int_part = int(dec_part)
        mantissa += str(int_part)
        dec_part -= int_part
        if dec_part == 0:
            break
    return mantissa + '0' * (length - len(mantissa))

def IEEE754_v2_tensor(numbers):
    signs = np.where(numbers < 0, 1, 0)
    numbers = np.abs(numbers)
    int_parts = np.floor(numbers).astype(int)
    dec_parts = numbers - int_parts
    int_bin_parts = np.array([bin(x).replace('0b', '') if x > 0 else '' for x in int_parts])
    mantissas = []
    exponents = []
    for i in range(len(numbers)):
        if int_parts[i] > 0:
            mantissa = int_bin_parts[i][1:] + fractional_to_bin(dec_parts[i], 23 - len(int_bin_parts[i][1:]))
            exponent = len(int_bin_parts[i]) - 1
        else:
            fraction_bin = fractional_to_bin(dec_parts[i], 50)
            first_one = fraction_bin.find('1')
            exponent = -(first_one + 1)
            mantissa = fraction_bin[first_one + 1:first_one + 24]
        mantissa = (mantissa + '0' * 23)[:23]
        mantissas.append(mantissa)
        exponents.append(exponent)
    exponents = np.array(exponents) + 127
    exponent_bits = np.array([bin(e).replace('0b', '').zfill(8) for e in exponents])
    ieee754_representations = np.array([
        str(signs[i]) + exponent_bits[i] + mantissas[i] for i in range(len(numbers))
    ])
    return ieee754_representations

def inv_IEEE754_tensor(num_IEEE_array):
    binary_matrix = np.array([list(num) for num in num_IEEE_array], dtype=int)
    signs = binary_matrix[:, 0]
    exponent_bits = binary_matrix[:, 1:9]
    exponents = np.dot(exponent_bits, 2 ** np.arange(7, -1, -1))
    mantissa_bits = binary_matrix[:, 9:].astype(float)
    powers = 2.0 ** np.arange(-1, -mantissa_bits.shape[1] - 1, -1, dtype=float)
    mantissas = np.dot(mantissa_bits, powers)
    normalized_mantissas = 1.0 + mantissas
    is_subnormal = (exponents == 0)
    exponents = np.where(is_subnormal, -126, exponents - 127)
    mantissas = np.where(is_subnormal, mantissas, normalized_mantissas)
    is_zero = (exponents == -127) & (mantissa_bits.sum(axis=1) == 0)
    numbers = mantissas * (2.0 ** exponents)
    numbers = np.where(is_zero, 0.0, numbers)
    numbers = np.where(signs == 1, -numbers, numbers)
    return numbers

def bitFLIP_v3_tensor(original_values, positions_list):
    original_values_np = original_values.cpu().detach().numpy()
    ieee_binary_strings = IEEE754_v2_tensor(original_values_np)
    flipped_binaries = []
    for i, positions in enumerate(positions_list):
        str_num = list(ieee_binary_strings[i])
        for position in positions:
            bit_position = 31 - position
            if bit_position == 1:
                bit_position = 0
            str_num[bit_position] = '0' if str_num[bit_position] == '1' else '1'
        if original_values_np[i] == 0:
            str_num = list('00000000000000000000000000000000')
        flipped_binaries.append("".join(str_num))
    flipped_values = inv_IEEE754_tensor(np.array(flipped_binaries))
    flipped_values_tensor = torch.tensor(flipped_values, dtype=original_values.dtype,
                                         device=original_values.device, requires_grad=True)
    return flipped_values_tensor