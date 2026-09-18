import unittest

import numpy as np
from scipy.sparse import csr_array

from virtual_brain import Connectome, ablate, lif_propagate, propagate_activity


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


if __name__ == "__main__":
    unittest.main()
