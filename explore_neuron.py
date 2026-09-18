import numpy as np
from scipy.io import loadmat


# ============================================================
# LOAD DATA
# ============================================================

print("Loading connectome...")

connectome = loadmat("data/connectome.mat")
annotations = loadmat("data/annotations.mat")

W = connectome["W"]

# TOT awalnya CSC sparse.
# Kita ubah ke CSR supaya pengambilan baris dengan getrow()
# lebih mudah dan tetap hemat RAM.
TOT = W["TOT"][0, 0].tocsr()

labels = annotations["labels"]
names = annotations["names"]

print("Connectome berhasil dimuat.")
print()


# ============================================================
# HELPER
# ============================================================

def matlab_string(value):
    """
    Mengubah object/string dari MATLAB menjadi Python string.
    """

    if isinstance(value, np.ndarray):
        value = value.squeeze()

        if value.size == 1:
            value = value.item()

    return str(value)


def get_name(field, index):
    """
    Mengambil nama anotasi neuron berdasarkan index.
    """

    label_id = int(
        labels[field][0, 0][index, 0]
    )

    # ID 0 berarti tidak diketahui
    if label_id == 0:
        return "Unknown"

    name_list = names[field][0, 0]

    if label_id > len(name_list):
        return f"Unknown (ID {label_id})"

    return matlab_string(
        name_list[label_id - 1]
    )


# ============================================================
# TARGET NEURON
# ============================================================

neuron = 90883


# ============================================================
# VALIDASI INDEX
# ============================================================

if neuron < 0 or neuron >= TOT.shape[0]:
    raise ValueError(
        f"Neuron index harus antara 0 dan {TOT.shape[0] - 1}"
    )


# ============================================================
# INFORMASI NEURON UTAMA
# ============================================================

print("=" * 60)
print(f"NEURON UTAMA : {neuron}")
print("=" * 60)

print()

print("=== ANNOTATION NEURON ===")

fields = [
    "flow",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "hemibrain",
    "hemilineage",
    "side",
    "nerve",
]

for field in fields:
    print(
        f"{field:12}: {get_name(field, neuron)}"
    )


# ============================================================
# FIND CONNECTIONS
# ============================================================

row = TOT[neuron].tocoo()

targets = row.col
weights = row.data


print()
print("=" * 60)
print("CONNECTION INFORMATION")
print("=" * 60)

print(
    f"Jumlah koneksi pada baris neuron : {len(targets)}"
)


# ============================================================
# SHOW FIRST 30 CONNECTIONS
# ============================================================

print()
print("=== 30 TARGET NEURON PERTAMA ===")

limit = min(30, len(targets))

for i in range(limit):

    target = targets[i]
    weight = weights[i]

    print(
        f"{i + 1:2}. "
        f"Neuron {target:6} "
        f"| weight = {weight}"
    )


# ============================================================
# ANNOTATION TARGET NEURONS
# ============================================================

print()
print("=" * 60)
print("ANNOTATION TARGET NEURON")
print("=" * 60)

annotation_limit = min(20, len(targets))

for target in targets[:annotation_limit]:

    print()
    print(f"Neuron {target}")
    print("-" * 40)

    for field in fields:

        print(
            f"  {field:12}: "
            f"{get_name(field, target)}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Neuron index       : {neuron}")
print(f"Total neuron       : {TOT.shape[0]}")
print(f"Connection found   : {len(targets)}")
print(f"Format matrix      : CSR sparse")

print()
print("Selesai.")