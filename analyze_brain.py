import numpy as np
from scipy.io import loadmat

# Load connectome
data = loadmat("data/connectome.mat")

# Ambil struktur W
W = data["W"]

# Ambil total konektivitas
TOT = W["TOT"][0, 0]

print("=== FLYWIRE CONNECTOME ===")
print("Ukuran matriks :", TOT.shape)
print("Jumlah neuron  :", TOT.shape[0])
print("Jumlah koneksi :", TOT.nnz)

# CSC:
# jumlah non-zero per kolom = koneksi masuk
in_degree = np.diff(TOT.indptr)

# Transpose untuk mendapatkan koneksi keluar
out_degree = np.diff(TOT.T.tocsc().indptr)

print()
print("Rata-rata koneksi masuk :", in_degree.mean())
print("Rata-rata koneksi keluar:", out_degree.mean())

# 10 neuron dengan koneksi keluar terbanyak
top_out = np.argsort(out_degree)[-10:][::-1]

print()
print("=== TOP 10 OUTGOING CONNECTIONS ===")

for neuron in top_out:
    print(
        f"Neuron {neuron:6d} "
        f"-> {out_degree[neuron]:5d} koneksi"
    )