import unittest

import numpy as np
from scipy.sparse import csr_array

from virtual_brain import (
    AblationResult,
    Connectome,
    SensoryExperimentResult,
    ablate,
    ablate_batch,
    lif_propagate,
    propagate_activity,
    run_sensory_experiment,
)


class VirtualBrainTests(unittest.TestCase):
    def test_propagation_uses_outgoing_weights(self):
        graph = csr_array(
            [
                [0, 2, 0],
                [0, 0, 2],
                [0, 0, 0],
            ],
            dtype=float,
        )

        history = propagate_activity(graph, [0], steps=2, threshold=1)

        np.testing.assert_array_equal(history[0], [0])
        np.testing.assert_array_equal(history[1], [1])
        np.testing.assert_array_equal(history[2], [2])

    def test_ablation_removes_input_and_output_edges(self):
        graph = csr_array(
            [
                [0, 1, 0],
                [2, 0, 3],
                [0, 4, 0],
            ],
            dtype=float,
        )

        result = ablate(graph, 1)

        self.assertEqual(result.nnz, 0)

    def test_propagation_is_deterministic(self):
        graph = csr_array([[0, 1], [0, 0]], dtype=float)

        first = propagate_activity(graph, [0], steps=3)
        second = propagate_activity(graph, [0], steps=3)

        self.assertEqual(len(first), len(second))
        for first_step, second_step in zip(first, second):
            np.testing.assert_array_equal(first_step, second_step)


class CoordinateTests(unittest.TestCase):
    def _make_brain(self, n: int) -> Connectome:
        matrix = csr_array((n, n), dtype=float)
        annotations = {"flow": tuple("Unknown" for _ in range(n))}
        coordinates = np.arange(n * 3, dtype=float).reshape(n, 3)
        return Connectome(matrix=matrix, annotations=annotations, coordinates=coordinates)

    def test_get_coordinates_returns_correct_rows(self):
        brain = self._make_brain(5)
        result = brain.get_coordinates([0, 2, 4])
        np.testing.assert_array_equal(result, brain.coordinates[[0, 2, 4]])

    def test_get_coordinates_raises_on_out_of_range(self):
        brain = self._make_brain(5)
        with self.assertRaises(IndexError):
            brain.get_coordinates([10])

    def test_coordinates_shape_matches_neuron_count(self):
        brain = self._make_brain(10)
        self.assertEqual(brain.coordinates.shape, (10, 3))


class LIFTests(unittest.TestCase):
    def _chain_graph(self, n: int) -> csr_array:
        """0 → 1 → 2 → … → n-1, weight 100 each."""
        row = list(range(n - 1))
        col = list(range(1, n))
        data = [100.0] * (n - 1)
        return csr_array((data, (row, col)), shape=(n, n), dtype=float)

    def test_lif_returns_one_state_per_step_plus_initial(self):
        graph = self._chain_graph(4)
        history = lif_propagate(graph, [0], steps=3)
        self.assertEqual(len(history), 4)

    def test_lif_spike_propagates_along_chain(self):
        graph = self._chain_graph(4)
        history = lif_propagate(
            graph, [0], steps=3,
            threshold=0.5, leak=0.9, weight_scale=0.02, refractory_steps=0,
        )
        # t=0: neuron 0 fires (initial)
        self.assertIn(0, history[0].active_indices)
        # t=1: neuron 1 should receive enough input to fire
        self.assertIn(1, history[1].active_indices)

    def test_lif_refractory_prevents_immediate_refiring(self):
        # Single neuron with self-loop, high weight
        graph = csr_array([[100.0]], dtype=float)
        history = lif_propagate(
            graph, [0], steps=3,
            threshold=0.5, leak=0.9, weight_scale=0.1, refractory_steps=2,
        )
        # Neuron fires at t=0; with refractory=2 it cannot fire at t=1 or t=2
        self.assertNotIn(0, history[1].active_indices)
        self.assertNotIn(0, history[2].active_indices)

    def test_lif_is_deterministic(self):
        graph = self._chain_graph(5)
        first = lif_propagate(graph, [0], steps=4)
        second = lif_propagate(graph, [0], steps=4)
        for s1, s2 in zip(first, second):
            np.testing.assert_array_equal(s1.active_indices, s2.active_indices)
            np.testing.assert_array_almost_equal(s1.potential, s2.potential)

    def test_lif_ablation_silences_downstream(self):
        graph = self._chain_graph(4)
        ablated_graph = ablate(graph, 0)
        history = lif_propagate(ablated_graph, [0], steps=3,
                                threshold=0.5, leak=0.9, weight_scale=0.02)
        # After ablation neuron 0 has no outgoing edges; only fires at t=0
        for state in history[1:]:
            self.assertNotIn(1, state.active_indices)


class BatchAblationTests(unittest.TestCase):
    def _chain_graph(self, n: int) -> csr_array:
        row = list(range(n - 1))
        col = list(range(1, n))
        data = [100.0] * (n - 1)
        return csr_array((data, (row, col)), shape=(n, n), dtype=float)

    def test_batch_returns_one_result_per_neuron(self):
        graph = self._chain_graph(5)
        results = ablate_batch(graph, neurons=[0, 1, 2], steps=3,
                               model="threshold", threshold=1.0)
        self.assertEqual(len(results), 3)

    def test_batch_result_fields_are_non_negative(self):
        graph = self._chain_graph(5)
        results = ablate_batch(graph, neurons=[0], steps=3,
                               model="threshold", threshold=1.0)
        r = results[0]
        self.assertGreaterEqual(r.total_spikes_normal, 0)
        self.assertGreaterEqual(r.total_spikes_ablated, 0)
        self.assertGreaterEqual(r.coverage_normal, 0.0)
        self.assertGreaterEqual(r.coverage_ablated, 0.0)

    def test_batch_spike_delta_equals_normal_minus_ablated(self):
        graph = self._chain_graph(6)
        results = ablate_batch(graph, neurons=[0, 1], steps=4,
                               model="threshold", threshold=1.0)
        for r in results:
            self.assertEqual(r.spike_delta,
                             r.total_spikes_normal - r.total_spikes_ablated)

    def test_batch_ablating_seed_reduces_activity(self):
        # Ablating neuron 0 in a chain should reduce downstream propagation
        graph = self._chain_graph(5)
        results = ablate_batch(graph, neurons=[0], steps=4,
                               model="threshold", threshold=1.0)
        r = results[0]
        self.assertGreater(r.total_spikes_normal, r.total_spikes_ablated)

    def test_batch_invalid_model_raises(self):
        graph = self._chain_graph(3)
        with self.assertRaises(ValueError):
            ablate_batch(graph, neurons=[0], steps=2, model="invalid")


class SensoryInterfaceTests(unittest.TestCase):
    def _make_brain(self, n: int, n_afferent: int, n_efferent: int) -> "Connectome":
        """Synthetic brain: first n_afferent are afferent, last n_efferent efferent."""
        matrix = csr_array((n, n), dtype=float)
        flow = (
            ["afferent"] * n_afferent
            + ["efferent"] * n_efferent
            + ["intrinsic"] * (n - n_afferent - n_efferent)
        )
        annotations = {
            "flow": tuple(flow),
            "super_class": tuple(["Unknown"] * n),
        }
        coordinates = np.zeros((n, 3), dtype=float)
        return Connectome(matrix=matrix, annotations=annotations, coordinates=coordinates)

    def test_sensory_experiment_returns_correct_step_count(self):
        brain = self._make_brain(10, n_afferent=2, n_efferent=2)
        result = run_sensory_experiment(
            brain, hops=0, steps=3, model="threshold", threshold=1.0
        )
        self.assertEqual(len(result.history), 4)  # t=0 … t=3

    def test_sensory_ids_are_all_afferent(self):
        brain = self._make_brain(10, n_afferent=3, n_efferent=2)
        result = run_sensory_experiment(
            brain, hops=0, steps=2, model="threshold", threshold=1.0
        )
        for nid in result.sensory_ids:
            self.assertEqual(brain.annotations["flow"][nid], "afferent")

    def test_motor_ids_are_all_efferent(self):
        brain = self._make_brain(10, n_afferent=2, n_efferent=3)
        result = run_sensory_experiment(
            brain, hops=0, steps=2, model="threshold", threshold=1.0
        )
        for nid in result.motor_ids:
            self.assertEqual(brain.annotations["flow"][nid], "efferent")

    def test_total_motor_spikes_non_negative(self):
        brain = self._make_brain(10, n_afferent=2, n_efferent=2)
        result = run_sensory_experiment(
            brain, hops=0, steps=3, model="threshold", threshold=1.0
        )
        self.assertGreaterEqual(result.total_motor_spikes(), 0)

    def test_no_afferent_raises(self):
        brain = self._make_brain(5, n_afferent=0, n_efferent=2)
        with self.assertRaises(ValueError):
            run_sensory_experiment(brain, hops=0, steps=2,
                                   model="threshold", threshold=1.0)


if __name__ == "__main__":
    unittest.main()
