# Virtual Fly Brain — Project Status

> Last updated: 2026-09-19

---

## Overview

Computational neuroscience prototype simulating *Drosophila melanogaster* neural dynamics using the **FlyWire connectome** dataset. The goal is to model sensory input → network propagation → motor output → behavioral readout.

---

## Dataset

| File | Description | Shape |
|------|-------------|-------|
| `data/connectome.mat` | Synaptic weight matrix (W.TOT) | 138 639 × 138 639, 15 091 983 edges |
| `data/annotations.mat` | Neuron metadata (9 fields) | 138 639 rows |
| `data/coordinates.mat` | 3-D neuron positions | 138 639 × 3 float64 |

**Key neuron counts**

| Flow | Count |
|------|-------|
| afferent (sensory input) | 18 668 |
| efferent (motor output) | 1 481 |
| intrinsic | 118 490 |

| Afferent super_class | Count |
|----------------------|-------|
| sensory | 16 351 |
| ascending | 2 317 |

| Efferent super_class | Count |
|----------------------|-------|
| descending | 1 299 |
| motor | 106 |
| endocrine | 76 |

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.13.12 (`.venv`) |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

---

## PRD Stages

| # | Stage | Status |
|---|-------|--------|
| 1 | Connectome loading & processing | ✅ Done |
| 2 | Neuron annotation mapping | ✅ Done |
| 3 | Sub-connectome extraction (N-hop, sparse) | ✅ Done |
| 4 | Neural dynamics — threshold + LIF models | ✅ Done |
| 5 | Activity propagation + 3-D visualization | ✅ Done |
| 6 | Sensory / motor interface | ✅ Done |
| 7 | VirtualFly — BehaviorState, action labels, turn_bias | ✅ Done |
| 8 | Batch ablation experiments | ✅ Done |
| 9 | Oscillation analysis | ✅ Done |
| — | Reproducibility — git + requirements.txt | ✅ Done |
| — | Multi-stimulus comparison | ✅ Done |
| — | Behavioral readout visualization | ✅ Done |

---

## File Structure

```
virtual-fly/
├── virtual_brain.py          # Core library (~700 lines)
├── run_propagation.py        # CLI: threshold vs LIF propagation
├── run_ablation_batch.py     # CLI: batch ablation of top-N neurons
├── run_sensory_experiment.py # CLI: sensory → network → motor simulation
├── run_virtual_fly.py        # CLI: full VirtualFly behavioral simulation
├── run_multi_stimulus.py     # CLI: multi-stimulus comparison (new)
├── visualize_activity.py     # 3-D scatter activity visualization
├── visualize_behavior.py     # Behavioral readout figure (5-panel, multi-condition)
├── test_virtual_brain.py     # Unit tests (25 tests, all passing)
├── requirements.txt          # Pinned dependencies
├── .gitignore
├── STATUS.md                 # ← this file
├── analyze_brain.py          # Exploratory script
├── explore_neuron.py         # Exploratory script (csr_array bug fixed)
├── inspect_annotations.py    # Exploratory script
├── neuron_info.py            # Exploratory script
└── data/
    ├── connectome.mat
    ├── annotations.mat
    └── coordinates.mat
```

---

## Core Library — `virtual_brain.py`

### Dataclasses

| Class | Description |
|-------|-------------|
| `Connectome` | Frozen: sparse matrix, annotations, coordinates |
| `LIFState` | LIF simulation state per timestep |
| `AblationResult` | Normal vs ablated spike comparison for one neuron |
| `SensoryExperimentResult` | Full sensory→motor experiment output |
| `BehaviorState` | Per-step behavioral readout (drives + action label) |
| `StimulusCondition` | Named stimulus config for multi-stimulus experiments |
| `MultiStimulusResult` | Side-by-side metrics across stimulus conditions |
| `OscillationResult` | Summary of transition rate and autocorrelation metrics per condition |

### Key Functions

| Function | Description |
|----------|-------------|
| `load_connectome()` | Load all three `.mat` files into a `Connectome` |
| `propagate_activity()` | Threshold-based activity propagation |
| `lif_propagate()` | Leaky Integrate-and-Fire propagation |
| `ablate()` | Remove a neuron's edges from the graph |
| `ablate_batch()` | Ablate top-N neurons, return ranked `AblationResult` list |
| `run_sensory_experiment()` | Afferent seed → sub-connectome → efferent tracking |
| `compare_stimuli()` | Run multiple `StimulusCondition`s and return `MultiStimulusResult` |
| `analyze_oscillation()` | Compute dominant period, transition rate, and autocorrelation across behavior time series |

### VirtualFly class

Wraps a `Connectome` into a fly agent. `run()` returns a list of `BehaviorState` per timestep with:
- `locomotion_drive` — descending efferent activity
- `left_drive` / `right_drive` — lateralized motor drive
- `turn_bias` = `right_drive − left_drive`
- `action` — string label: `"walk"`, `"turn_left"`, `"turn_right"`, `"rest"`, `"modulated"`

---

## LIF Model Calibration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `threshold` | 0.5 | Firing threshold |
| `leak` | 0.9 | Membrane potential decay per step |
| `weight_scale` | 0.04 | Calibrated from p95 edge weight ≈ 13; 0.5/13 ≈ 0.038 |
| `refractory_steps` | 2 | Prevents immediate re-firing |

---

## Tests

```
python -m unittest -v test_virtual_brain.py
```

32 tests across 7 test classes — all passing:

| Class | Coverage |
|-------|----------|
| `VirtualBrainTests` | Propagation, ablation edges |
| `CoordinateTests` | Coordinate loading, shape |
| `LIFTests` | LIF firing, decay, refractory period |
| `BatchAblationTests` | Ablation ranking, delta computation |
| `SensoryInterfaceTests` | Sensory filter, motor tracking |
| `VirtualFlyTests` | BehaviorState fields, action labels, drives |
| `MultiStimulusTests` | compare_stimuli, metrics, condition names |
| `OscillationTests` | period, transition rate, autocorrelation analysis |

---

## Git History

| Commit | Message |
|--------|---------|
| `765bff1` | Update STATUS.md: add visualize_behavior.py entry |
| `6c14236` | Add visualize_behavior.py: multi-condition behavioral readout (5-panel figure) |
| `bf3cce3` | Update STATUS.md: multi-stimulus comparison complete with sample results |
| `324c1d9` | Add multi-stimulus comparison: StimulusCondition, MultiStimulusResult, compare_stimuli, CLI, tests |
| `69aaa0a` | Improve sensory_filter error: show valid super_class values when filter matches nothing |
| (prior) | Add VirtualFly behavioral simulation (Stage 7) |
| (prior) | Add batch ablation CLI and tests (Stage 8) |
| (prior) | Add sensory/motor interface (Stage 6) |
| (prior) | Initial connectome core + LIF model |

---

## Completed Features (beyond PRD v0.1)

### Multi-Stimulus Comparison ✅

**Goal:** Compare `sensory` vs `ascending` afferent inputs under identical simulation conditions.

**Metrics compared:**
- `seed_neurons` — number of input neurons
- `total_spikes` — total network activity
- `motor_spikes` — efferent neuron activations
- `mean_locomotion` — average locomotion drive
- `mean_turn_bias` — average directional bias
- `dominant_action` — most frequent behavior label

**Sample output (8 steps, LIF):**

| condition | seeds | total_spikes | motor_spikes | mean_loco | mean_bias | dominant_action |
|-----------|-------|-------------|--------------|-----------|-----------|-----------------|
| sensory | 50 | 2702 | 41 | 0.3035 | +0.2550 | turn_right |
| ascending | 50 | 7765 | 112 | 0.2918 | +0.2987 | turn_right |

Key observations:
- `ascending` neurons drive ~2.9× more total network activity than `sensory`
- `ascending` recruits ~2.7× more motor spikes
- Both converge on `turn_right` as dominant action
- `sensory` shows more oscillation between `walk_forward` and `turn_right` per step

```bash
python run_multi_stimulus.py --steps 8 --model lif
```

---

### Behavioral Readout Visualization ✅

**File:** `visualize_behavior.py`

5-panel matplotlib figure comparing all stimulus conditions:

| Panel | Content |
|-------|---------|
| Row 1 | Active neuron count per timestep |
| Row 2 | Motor/efferent spike count per timestep |
| Row 3 | Locomotion + left/right/endocrine drives + turn bias (dual axis) |
| Row 4 | Action label timeline (color-coded spans) |
| Row 5 | Action distribution bar chart (all conditions side-by-side) |

```bash
python visualize_behavior.py --steps 12 --model lif --save output_behavior.png
```

---

### Oscillation Analysis ✅

**Goal:** quantify periodic behavior in the fly’s action sequence and continuous drives.

**Metrics computed:**
- `transition_rate` — how often actions switch between adjacent timesteps
- `dominant_period` — recurrence period of the turn_bias signal
- `peak_autocorr_turn_bias` — strongest positive autocorrelation in directional bias
- `peak_autocorr_locomotion` — strongest positive autocorrelation in locomotion drive

**Implementation:** `analyze_oscillation()` in `virtual_brain.py` consumes the real `SensoryExperimentResult` stream and summarizes each condition in an `OscillationResult`.

**Validation:** verified with the full project suite using `python -m unittest -v test_virtual_brain.py` — 32 tests pass.

---

## Next Steps

| Priority | Task | Notes |
|----------|------|-------|
| High | Export to CSV | Per-step metrics → CSV for downstream statistical analysis |
| High | Parameter sensitivity sweep | Validate that `sensory` vs `ascending` trends persist across threshold/leak/weight settings |
| Medium | Ablation × stimulus refinement | Rank neurons by behavior disruption rather than raw spike loss alone |
| Medium | Add more stimulus types | Verify `class` / `sub_class` labels with `inspect_annotations.py` to define `visual`, `tactile`, etc. |
| Low | 3-D anatomical overlay | Color neurons by condition in `visualize_activity.py` |
| Low | Remote git identity | `git config --global user.name / user.email` to fix committer name warning |

---

## Known Issues / Notes

- `super_class = "visual"` does not exist in the dataset; valid values are `"sensory"` and `"ascending"` for afferent neurons.
- `TOT[neuron]` returns `coo_array` — use `.tocoo().col` or `.tocsr()` row slicing, not `.getrow()`.
- Large sub-connectome extraction with `hops > 1` can be slow on the full 138 K graph.
