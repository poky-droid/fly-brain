import numpy as np
from scipy.io import loadmat


# =========================
# LOAD DATA
# =========================

connectome = loadmat("data/connectome.mat")
annotations = loadmat("data/annotations.mat")

W = connectome["W"]
TOT = W["TOT"][0, 0]

labels = annotations["labels"]
names = annotations["names"]


# =========================
# HELPER
# =========================

def matlab_string(value):
    """
    Mengubah object/string MATLAB menjadi string Python.
    """
    if isinstance(value, np.ndarray):
        value = value.squeeze()

        if value.size == 1:
            value = value.item()

    return str(value)


def get_name(field, index):
    """
    Mengubah ID label menjadi nama.
    """
    label_id = int(labels[field][0, 0][index, 0])

    if label_id == 0:
        return "Unknown"

    name_list = names[field][0, 0]

    # MATLAB biasanya menggunakan index mulai dari 1
    if label_id > len(name_list):
        return f"Unknown (ID {label_id})"

    return matlab_string(name_list[label_id - 1])


# =========================
# NEURON INFO
# =========================

def neuron_info(index):

    if index < 0 or index >= TOT.shape[0]:
        raise ValueError(
            f"Index harus antara 0 dan {TOT.shape[0] - 1}"
        )

    print("=" * 50)
    print(f"NEURON INDEX : {index}")
    print("=" * 50)

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
            f"{field:12}: {get_name(field, index)}"
        )


# =========================
# TEST
# =========================

neuron_info(90883)