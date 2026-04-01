import tenseal as ts

try:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_modulus_bits=[60,40,40,60])
    print("coeff_modulus_bits works")
except TypeError as e:
    print(f"coeff_modulus_bits failed: {e}")

try:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_lengths=[60,40,40,60])
    print("coeff_mod_bit_lengths works")
except TypeError as e:
    print(f"coeff_mod_bit_lengths failed: {e}")

try:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_moduli=[60,40,40,60])
    print("coeff_moduli works")
except TypeError as e:
    print(f"coeff_moduli failed: {e}")