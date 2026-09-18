# PRD — Virtual Fly Brain

**Product Requirements Document**

| Field | Detail |
|---|---|
| Nama Project | Virtual Fly Brain |
| Versi | 0.1 |
| Status | Prototype / Research |
| Platform | macOS / Linux |
| Bahasa | Python |
| Target Awal | Local research prototype |
| Dataset Utama | FlyWire-derived connectome |
| Primary Compute | MacBook Air M2, 16 GB RAM |

---

# 1. Executive Summary

Virtual Fly Brain adalah project research/computational neuroscience yang bertujuan membangun representasi komputasional dari jaringan saraf *Drosophila melanogaster* menggunakan data connectome.

Project tidak bertujuan langsung membuat replika biologis otak lalat secara sempurna.

Tahap awal berfokus pada:

1. Memproses data connectome.
2. Memetakan neuron dengan annotation biologis.
3. Membentuk sub-connectome.
4. Memodelkan dinamika neuron.
5. Mensimulasikan propagasi aktivitas.
6. Membuat interface sensorik dan motorik.
7. Membuat virtual fly sederhana.
8. Melakukan eksperimen neuron ablation.

Konsep utama:

```text
CONNECTOME
    ↓
NEURON NETWORK
    ↓
NEURAL DYNAMICS
    ↓
SENSORY INPUT
    ↓
MOTOR OUTPUT
    ↓
VIRTUAL FLY
    ↓
EXPERIMENT