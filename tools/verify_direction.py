import numpy as np
from scipy.io import loadmat


# ============================================================
# LOAD DATA
# ============================================================

connectome = loadmat("data/connectome.mat")

W = connectome["W"]

TOT = W["TOT"][0, 0].tocsr()


# ============================================================
# TEST NEURONS
# ============================================================

neuron_a = 90883
neuron_b = 37


# ============================================================
# CHECK CONNECTION A → B
# ============================================================

weight_a_to_b = TOT[neuron_a, neuron_b]


# ============================================================
# CHECK CONNECTION B → A
# ============================================================

weight_b_to_a = TOT[neuron_b, neuron_a]


# ============================================================
# RESULT
# ============================================================

print("=" * 60)
print("CONNECTION DIRECTION TEST")
print("=" * 60)

print()

print(
    f"Matrix[{neuron_a}, {neuron_b}] "
    f"= {weight_a_to_b}"
)

print(
    f"Matrix[{neuron_b}, {neuron_a}] "
    f"= {weight_b_to_a}"
)

print()

if weight_a_to_b != 0:

    print(
        f"Connection exists: "
        f"{neuron_a} → {neuron_b}"
    )

    print(
        f"Weight = {weight_a_to_b}"
    )

else:

    print(
        f"No connection found: "
        f"{neuron_a} → {neuron_b}"
    )


print()

if weight_b_to_a != 0:

    print(
        f"Connection exists: "
        f"{neuron_b} → {neuron_a}"
    )

    print(
        f"Weight = {weight_b_to_a}"
    )

else:

    print(
        f"No connection found: "
        f"{neuron_b} → {neuron_a}"
    )