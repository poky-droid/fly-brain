"""Reusable sparse connectome utilities for the Virtual Fly prototype."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_array, issparse


DEFAULT_FIELDS = (
    "flow",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "hemibrain",
    "hemilineage",
    "side",
    "nerve",
)


@dataclass(frozen=True)
class Connectome:
    """Sparse connectome and decoded annotation data."""

    matrix: csr_array
    annotations: dict[str, tuple[str, ...]]
    coordinates: np.ndarray  # shape (N, 3), float64; nanometers in FlyWire space

    @property
    def neuron_count(self) -> int:
        return self.matrix.shape[0]

    def get_coordinates(self, neuron_ids: Iterable[int]) -> np.ndarray:
        """Return the (x, y, z) positions for a set of neuron IDs."""
        ids = _validate_indices(neuron_ids, self.neuron_count)
        return self.coordinates[ids]

    def annotation_matches(self, field: str, value: str) -> np.ndarray:
        """Return zero-based neuron IDs whose annotation equals ``value``."""
        if field not in self.annotations:
            raise KeyError(f"Unknown annotation field: {field}")
        return np.flatnonzero(np.asarray(self.annotations[field]) == value)

    def induced_subgraph(
        self,
        seeds: Iterable[int],
        hops: int = 1,
    ) -> tuple[np.ndarray, csr_array]:
        """Return neuron IDs and the sparse graph induced by their neighborhood."""
        if hops < 0:
            raise ValueError("hops must be non-negative")

        selected = _validate_indices(seeds, self.neuron_count)
        for _ in range(hops):
            if selected.size == 0:
                break
            neighbors = self.matrix[selected].nonzero()[1]
            selected = np.unique(np.concatenate((selected, neighbors)))

        return selected, self.matrix[selected][:, selected].tocsr()


def load_connectome(
    connectome_path: str | Path = "data/connectome.mat",
    annotations_path: str | Path = "data/annotations.mat",
    coordinates_path: str | Path = "data/coordinates.mat",
) -> Connectome:
    """Load the MATLAB dataset without densifying the connectome."""
    connectome_data = loadmat(connectome_path)
    annotation_data = loadmat(annotations_path)
    coordinate_data = loadmat(coordinates_path)

    matrix = connectome_data["W"]["TOT"][0, 0]
    if not issparse(matrix):
        raise TypeError("W.TOT must be a SciPy sparse matrix")
    matrix = matrix.tocsr()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"W.TOT must be square, got {matrix.shape}")

    labels = annotation_data["labels"]
    names = annotation_data["names"]
    annotations: dict[str, tuple[str, ...]] = {}
    for field in labels.dtype.names or ():
        label_values = labels[field][0, 0].reshape(-1)
        name_values = names[field][0, 0].reshape(-1)
        if label_values.size != matrix.shape[0]:
            raise ValueError(
                f"Annotation field {field!r} has {label_values.size} values; "
                f"expected {matrix.shape[0]}"
            )
        annotations[field] = tuple(
            _decode_label(label_id, name_values) for label_id in label_values
        )

    coordinates = coordinate_data["coor"]
    if coordinates.shape != (matrix.shape[0], 3):
        raise ValueError(
            f"coordinates must have shape ({matrix.shape[0]}, 3), "
            f"got {coordinates.shape}"
        )

    return Connectome(matrix=matrix, annotations=annotations, coordinates=coordinates)


def propagate_activity(
    graph: csr_array,
    initial_active: Iterable[int],
    steps: int,
    threshold: float = 1.0,
    decay: float = 0.0,
) -> list[np.ndarray]:
    """Simulate binary activity using weighted incoming signals.

    Each timestep sums outgoing weights from currently active neurons. A neuron
    fires when its signal exceeds ``threshold``; ``decay`` retains part of the
    previous signal between timesteps.
    """
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not 0 <= decay < 1:
        raise ValueError("decay must be in the range [0, 1)")

    active = np.zeros(graph.shape[0], dtype=bool)
    initial = _validate_indices(initial_active, graph.shape[0])
    active[initial] = True
    signal = np.zeros(graph.shape[0], dtype=float)
    history = [np.flatnonzero(active)]

    for _ in range(steps):
        signal = decay * signal + np.asarray(active @ graph).reshape(-1)
        active = signal >= threshold
        history.append(np.flatnonzero(active))

    return history


@dataclass
class LIFState:
    """Snapshot of membrane potentials and refractory counters at one timestep."""

    step: int
    potential: np.ndarray      # shape (N,) float64
    spiked: np.ndarray         # shape (N,) bool — neurons that fired this step
    refractory: np.ndarray     # shape (N,) int — remaining refractory steps

    @property
    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.spiked)


def lif_propagate(
    graph: csr_array,
    initial_active: Iterable[int],
    steps: int,
    threshold: float = 0.5,
    leak: float = 0.9,
    reset_potential: float = 0.0,
    weight_scale: float = 0.04,
    refractory_steps: int = 2,
) -> list[LIFState]:
    """Simulate Leaky Integrate-and-Fire dynamics over a sparse subgraph.

    At each timestep:
        V[t] = V[t-1] * (1 - leak) + weight_scale * (spikes[t-1] @ graph)
    A neuron fires when V >= threshold; V is then reset to reset_potential and the
    neuron enters a refractory period during which it cannot fire again.

    Parameters
    ----------
    graph:
        Sparse CSR weight matrix. Rows are source neurons, columns are targets.
    initial_active:
        Local indices (within the subgraph) of neurons forced to fire at t=0.
    steps:
        Number of timesteps to simulate after t=0.
    threshold:
        Membrane potential required to fire.
    leak:
        Fraction of potential retained each timestep (0 = no memory, 1 = no leak).
    reset_potential:
        Membrane potential after a spike.
    weight_scale:
        Multiplier applied to synaptic weights before accumulation. Use to
        normalise the large integer weights in the FlyWire dataset.
    refractory_steps:
        How many subsequent steps a neuron cannot fire after spiking.
    """
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    if refractory_steps < 0:
        raise ValueError("refractory_steps must be non-negative")

    n = graph.shape[0]
    initial = _validate_indices(initial_active, n)

    V = np.zeros(n, dtype=float)
    refrac = np.zeros(n, dtype=int)
    spiked = np.zeros(n, dtype=bool)
    spiked[initial] = True
    V[initial] = threshold
    refrac[initial] = refractory_steps  # initial spikes also enter refractory

    history: list[LIFState] = [
        LIFState(
            step=0,
            potential=V.copy(),
            spiked=spiked.copy(),
            refractory=refrac.copy(),
        )
    ]

    for t in range(1, steps + 1):
        # Synaptic input from neurons that spiked last step
        I = weight_scale * np.asarray(spiked @ graph).reshape(-1)

        # Leak + integrate (refractory neurons receive input but cannot fire)
        V = V * (1.0 - leak) + I

        # Determine which neurons fire
        can_fire = refrac == 0
        new_spikes = can_fire & (V >= threshold)

        # Reset spiking neurons and start refractory countdown
        V[new_spikes] = reset_potential
        refrac[new_spikes] = refractory_steps
        refrac[~new_spikes] = np.maximum(0, refrac[~new_spikes] - 1)
        spiked = new_spikes

        history.append(
            LIFState(
                step=t,
                potential=V.copy(),
                spiked=spiked.copy(),
                refractory=refrac.copy(),
            )
        )

    return history


def ablate(graph: csr_array, neuron: int) -> csr_array:
    """Return a copy of ``graph`` with one neuron's input and output removed."""
    _validate_indices([neuron], graph.shape[0])
    result = graph.copy().tolil()
    result[neuron, :] = 0
    result[:, neuron] = 0
    return result.tocsr()


@dataclass(frozen=True)
class AblationResult:
    """Metrics for a single neuron ablation experiment."""

    neuron_id: int
    total_spikes_normal: int
    total_spikes_ablated: int
    first_spike_step_normal: int   # -1 if never active
    first_spike_step_ablated: int  # -1 if never active
    coverage_normal: float
    coverage_ablated: float

    @property
    def spike_delta(self) -> int:
        return self.total_spikes_normal - self.total_spikes_ablated

    @property
    def coverage_delta(self) -> float:
        return self.coverage_normal - self.coverage_ablated


def ablate_batch(
    graph: csr_array,
    neurons: Iterable[int],
    steps: int,
    model: str = "lif",
    **model_kwargs,
) -> list[AblationResult]:
    """Run one ablation experiment per neuron and return comparative metrics.

    Parameters
    ----------
    graph:
        Sparse CSR subgraph.
    neurons:
        Local indices of neurons to ablate one at a time.
    steps:
        Simulation timesteps per run.
    model:
        ``"lif"`` (default) or ``"threshold"``.
    **model_kwargs:
        Forwarded to ``lif_propagate`` or ``propagate_activity``.
    """
    targets = _validate_indices(neurons, graph.shape[0])
    if model not in ("lif", "threshold"):
        raise ValueError(f"model must be 'lif' or 'threshold', got {model!r}")

    n = graph.shape[0]

    def _run(g: csr_array, seed: int) -> tuple[int, int, float]:
        if model == "lif":
            history = lif_propagate(g, [seed], steps=steps, **model_kwargs)
            active_per_step = [s.active_indices for s in history]
        else:
            active_per_step = propagate_activity(g, [seed], steps=steps, **model_kwargs)

        total = sum(len(a) for a in active_per_step)
        ever_active: set[int] = set()
        first_step = -1
        for t, a in enumerate(active_per_step):
            ever_active.update(a.tolist())
            if first_step == -1 and len(a) > 0:
                first_step = t
        coverage = len(ever_active) / n if n > 0 else 0.0
        return total, first_step, coverage

    results: list[AblationResult] = []
    for neuron in targets:
        total_n, first_n, cov_n = _run(graph, int(neuron))
        ablated_graph = ablate(graph, int(neuron))
        total_a, first_a, cov_a = _run(ablated_graph, int(neuron))
        results.append(
            AblationResult(
                neuron_id=int(neuron),
                total_spikes_normal=total_n,
                total_spikes_ablated=total_a,
                first_spike_step_normal=first_n,
                first_spike_step_ablated=first_a,
                coverage_normal=cov_n,
                coverage_ablated=cov_a,
            )
        )

    return results


def _validate_indices(indices: Iterable[int], size: int) -> np.ndarray:
    values = np.asarray(list(indices), dtype=int)
    if values.size and (values.min() < 0 or values.max() >= size):
        raise IndexError(f"Neuron index must be between 0 and {size - 1}")
    return np.unique(values)


def _decode_label(label_id: int, names: np.ndarray) -> str:
    label_id = int(label_id)
    if label_id == 0:
        return "Unknown"
    if label_id < 0 or label_id > names.size:
        return f"Unknown (ID {label_id})"
    value = names[label_id - 1]
    if isinstance(value, np.ndarray):
        value = value.squeeze()
        if value.size == 1:
            value = value.item()
    return str(value).strip()
