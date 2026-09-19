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
| 10 | Sensitivity sweep (LIF parameter scan) | ✅ Done |
| 11 | Sensitivity analysis (correlation + effect size + ranking) | ✅ Done |
| 12 | VirtualFly V1 — 2-D body, environment, motor loop | ✅ Done |
| 13 | FlyWireBrain adapter — persistent LIF in closed loop | ✅ Done |
| 14 | FlyWire sensory validation — left / right / front stimuli | ✅ Done |
| 15 | Annotation-driven sensory lateralization via `nerve` labels | ✅ Done |
| 16 | Motor population diagnostic and left-nerve normalization fix | ✅ Done |
| 17 | Temporal dynamics validation — constant and switching stimuli | ✅ Done |
| 18 | Behavioral scenario evaluation — light, obstacle, lateral/front stimuli | ✅ Done |
| 19 | Collision and locomotion diagnosis | ✅ Done |
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
├── run_multi_stimulus.py     # CLI: multi-stimulus comparison
├── sensitivity_sweep.py      # Reproducible sweep over threshold/leak/weight_scale/refractory_steps
├── analyze_sensitivity.py    # Correlation + effect-size + ranking summary across sweep results
├── export_results.py         # Tabular CSV export for condition-by-timestep results
├── run_fly_simulation.py     # CLI: closed-loop 2-D VirtualFly trajectory
├── validate_flywire_stimuli.py # CLI: compare left, right, and front FlyWire responses
├── validate_temporal_dynamics.py # CLI: temporal raw-spike diagnostics across six protocols
├── validate_behavioral_scenarios.py # CLI: scenario metrics and trajectory CSVs
├── validate_directional_behavior.py # CLI: closed-loop left/right/front evaluation
├── virtual_fly/              # V1 body, environment, FlyWire brain, simulation, plotting
├── test_virtual_fly_v1.py    # V1 body/environment/FlyWire regression tests
├── visualize_activity.py     # 3-D scatter activity visualization
├── visualize_behavior.py     # Behavioral readout figure (5-panel, multi-condition)
├── test_virtual_brain.py     # Core unit tests (34 tests, all passing)
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
| `SensitivitySweep` | Reproducible output of parameter combinations and per-condition metrics |

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
| `sensitivity_sweep.py` | Sweep threshold/leak/weight_scale/refractory_steps and export one row per parameter set |
| `analyze_sensitivity.py` | Rank the sensitivity of outputs by correlation magnitude and simple effect size |

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

43 tests across the core and V1 suites — all passing:

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
| `CSVExportTests` | CSV export for multi-stimulus + oscillation objects |
| `VirtualFlyV1Tests` | 2-D body movement, light sensing, and closed-loop trajectory |
| `FlyWireBrain` adapter | Persistent LIF state, sensory-bank encoding, motor decoding, reset determinism |
| `Behavioral scenarios` | Light approach, obstacle collision, and left/right/front trajectory metrics |

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

### FlyWireBrain Closed-Loop Adapter ✅

**File:** `virtual_fly/brain.py`

The 2-D simulator now uses the FlyWire connectome by default through a stepwise
`FlyWireBrain` adapter. Each timestep performs:

```text
environment sensors
    ↓
afferent sensory-bank encoding
    ↓
persistent LIF propagation
    ↓
descending / lateral / endocrine motor decoding
    ↓
body action and position update
```

The adapter preserves membrane potentials and refractory counters between
timesteps and supports `reset()` for deterministic replay. The real dataset
smoke test loads the connectome and produces `walk_forward` and `turn_right`
actions in the trajectory CLI:

```bash
python run_fly_simulation.py --steps 40
```

The adapter now selects balanced afferent populations using the dataset's
`nerve` annotations (`left` / `right`) before constructing the sub-connectome.
The deterministic split remains only as a fallback for synthetic or unlabeled
connectomes. The mapping source and selected bank sizes are printed by the
stimulus validation CLI.

### FlyWire Stimulus Validation ✅

**File:** `validate_flywire_stimuli.py`

The validation holds the fly pose fixed, resets the brain for each condition,
and compares three light positions:

```bash
python validate_flywire_stimuli.py --steps 6
```

The real-connectome run produced distinct sensory inputs and distinct motor
time series for `left`, `right`, and `front`. The annotation-driven mapping
removed the arbitrary ID-order split, but all three conditions still decoded a
stronger right motor drive than left motor drive. This validates
sensory-to-neural-to-motor sensitivity but does not establish biological
left/right symmetry; the remaining asymmetry is now a connectome/decoder
response to investigate.

### Motor Population Diagnostic ✅

The persistent zero left-drive bug was traced to a label mismatch: the dataset
decodes the annotation as `nerve = "left"`, while the decoder searched for
`"left "` with a trailing space. After normalization, the default real-data
subgraph contains both populations:

```text
left_local=56  right_local=41
```

Raw isolated protocols now show both left and right motor populations firing;
the observed 3-step rhythm remains a LIF/subgraph dynamical property to study,
not a consequence of an empty left decoder population.

### Temporal Dynamics Validation ✅

**File:** `validate_temporal_dynamics.py`

Six controlled protocols were measured for 100 timesteps:

- `constant_left`
- `constant_right`
- `constant_front`
- `stimulus_none` — left stimulus for 10 steps, then zero input
- `left_to_right`
- `right_to_left`

Each protocol records total active neurons, left/right motor spikes,
descending spikes, locomotion drive, motor drives, and action labels. The
current result shows a repeatable three-phase motor pattern under constant
input. The same pattern persists after the input switches to zero, indicating
that the observed periodicity is generated by recurrent subgraph/LIF state,
not only by the instantaneous sensory frame. Switching left/right inputs
changes the transition response, but does not immediately extinguish the
ongoing neural rhythm.

Run the diagnostic with:

```bash
python validate_temporal_dynamics.py --steps 100
```

This milestone measures the temporal behavior without modifying or claiming
to correct the LIF dynamics. Parameter, subgraph-depth, and recurrent-state
comparisons remain the next calibration work.

### Behavioral Scenario Evaluation ✅

**File:** `validate_behavioral_scenarios.py`

Five 100-timestep closed-loop scenarios now produce reproducible metrics and
trajectory CSVs:

- `light_source`
- `obstacle`
- `left_stimulus`
- `right_stimulus`
- `front_stimulus`

Measured outputs include initial/final/minimum light distance, final position
and heading, mean locomotion, mean turn bias, turn count, and collision count.
The evaluator writes:

```text
results/behavioral_scenarios/scenario_summary.csv
results/behavioral_scenarios/trajectory.csv
```

The current run showed the light scenario ending closer to its source, distinct
left/right/front trajectories, and high collision counts in the obstacle
scenario. These are behavioral measurements only; no LIF or motor parameters
were changed to improve the result.

The Milestone 17 obstacle baseline is preserved at:

```text
results/behavioral_scenarios/baseline_milestone_17.csv
```

### Directional Behavioral Evaluation ✅

**File:** `validate_directional_behavior.py`

The focused 100-step closed-loop evaluation compares `left`, `right`, and
`front` light stimuli while recording the complete requested/applied action,
sensor, motor, bias, movement, and collision trajectory. It writes:

```text
results/directional_behavior/summary.csv
results/directional_behavior/trajectory.csv
```

The current baseline shows distinct sensor means and action distributions:

```text
left:  turn_left=11, turn_right=10, walk_forward=10, rest=68
right: turn_right=13, turn_left=11, walk_forward=10, rest=66
front: walk_forward=30, turn_right=1, rest=69
```

Mean left/right motor drives are close in this long closed-loop run, so the
current evidence supports differentiated trajectories/actions more strongly
than a large mean motor-bias effect. This is an observation baseline; no brain,
LIF, decoder, or collision parameters were changed.

### Collision and Locomotion Diagnosis ✅

The evaluator now records both `collision_rate` and `movement_fraction`, and
each trajectory row distinguishes the brain's `requested_action` from the
post-collision `action` that was applied to the body.

The 100-step obstacle run measured:

```text
collision_rate    = 0.910
movement_fraction = 0.090
```

The first collision occurred at step 9. The FlyWire output continued to request
`walk_forward`, while collision handling applied `rest` and restored the pose.
This identifies the obstacle result as a trapped-body/locomotion interaction,
not an actively moving fly that merely collides often. No brain or LIF
parameters were changed; obstacle avoidance is now a separate behavioral
controller question for the next milestone.

---

### Oscillation Analysis ✅

**Goal:** quantify periodic behavior in the fly’s action sequence and continuous drives.

**Metrics computed:**
- `transition_rate` — how often actions switch between adjacent timesteps
- `dominant_period` — recurrence period of the turn_bias signal
- `peak_autocorr_turn_bias` — strongest positive autocorrelation in directional bias
- `peak_autocorr_locomotion` — strongest positive autocorrelation in locomotion drive

**Implementation:** `analyze_oscillation()` in `virtual_brain.py` consumes the real `SensoryExperimentResult` stream and summarizes each condition in an `OscillationResult`.

**Validation:** verified with the full project suite using `python -m unittest -v test_virtual_brain.py` — 34 tests pass.

---

### Parameter Sensitivity Sweep + Analysis ✅

**Goal:** determine which LIF parameters most strongly shape observable behavior, beyond a single default configuration.

**Files:**
- `sensitivity_sweep.py` — reproducible sweep over `threshold`, `leak`, `weight_scale`, and `refractory_steps`
- `analyze_sensitivity.py` — computes parameter→output correlation, average output shifts, simple effect size, and ranking by absolute correlation magnitude

**Outputs tracked per row:**
- `condition`
- `threshold`, `leak`, `weight_scale`, `refractory_steps`
- `total_spikes`, `motor_spikes`, `mean_locomotion`, `mean_turn_bias`
- `transition_rate`, `dominant_period`

**Interpretation method:**
- compare `pearson_r` for each parameter-output pair
- compute change in mean output across the parameter range
- estimate simple effect size using the high-vs-low contrast
- rank parameters by `|r|` for each output metric

**Current observed pattern:**
- `weight_scale` and `refractory_steps` are the strongest drivers for locomotion/bias outcomes
- `threshold` has a notable effect on transition rate
- `leak` shows comparatively weak sensitivity in this narrowed sweep

---

## Next Steps

| Priority | Task | Notes |
|----------|------|-------|
| High | Expanded parameter sweep | Increase grid resolution for paper-quality sensitivity analysis |
| High | Behavior-guided ablation | Rank neurons by behavior disruption rather than raw spike loss alone |
| Medium | Add more stimulus types | Verify `class` / `sub_class` labels with `inspect_annotations.py` to define `visual`, `tactile`, etc. |
| Medium | CSV export packaging | Save per-condition CSVs under `results/` with clear naming convention |
| Low | 3-D anatomical overlay | Color neurons by condition in `visualize_activity.py` |
| Low | Remote git identity | `git config --global user.name / user.email` to fix committer name warning |

---

## Known Issues / Notes

- `super_class = "visual"` does not exist in the dataset; valid values are `"sensory"` and `"ascending"` for afferent neurons.
- `TOT[neuron]` returns `coo_array` — use `.tocoo().col` or `.tocsr()` row slicing, not `.getrow()`.
- Large sub-connectome extraction with `hops > 1` can be slow on the full 138 K graph.
