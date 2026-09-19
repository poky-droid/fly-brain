"""Adapter that exposes the FlyWire connectome as a stepwise brain."""

from __future__ import annotations

import numpy as np

from virtual_brain import BehaviorState, VirtualFly as ConnectomeVirtualFly

from .controller import MotorCommand


class FlyWireBrain:
    """Stepwise FlyWire brain adapter for the closed-loop simulator.

    Afferent neurons are assigned to input banks using their normalized
    ``nerve`` annotation when available. Sensor intensity controls how many
    neurons in each biologically labeled bank spike at a step.
    """

    def __init__(
        self,
        brain,
        max_sensory: int = 50,
        hops: int = 1,
        threshold: float = 0.5,
        leak: float = 0.9,
        weight_scale: float = 0.04,
        refractory_steps: int = 2,
    ) -> None:
        self.connectome = brain
        (
            sensory_ids,
            self._left_sensory_ids,
            self._right_sensory_ids,
            self.mapping_source,
        ) = self._select_sensory_ids(brain, max_sensory)
        self.model = ConnectomeVirtualFly(
            brain,
            max_sensory=max_sensory,
            hops=hops,
            sensory_ids=sensory_ids,
        )
        self.threshold = threshold
        self.leak = leak
        self.weight_scale = weight_scale
        self.refractory_steps = refractory_steps
        self._sensory_local = np.flatnonzero(
            np.isin(self.model._subgraph_ids, self.model._sensory_ids)
        )
        self._left_input = np.flatnonzero(
            np.isin(self.model._subgraph_ids, self._left_sensory_ids)
        )
        self._right_input = np.flatnonzero(
            np.isin(self.model._subgraph_ids, self._right_sensory_ids)
        )
        if self.mapping_source == "fallback":
            midpoint = len(self._sensory_local) // 2
            self._left_input = self._sensory_local[:midpoint]
            self._right_input = self._sensory_local[midpoint:]
        self.reset()

    @staticmethod
    def _select_sensory_ids(
        brain,
        max_sensory: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        afferent = brain.annotation_matches("flow", "afferent")
        if "nerve" not in brain.annotations:
            selected = afferent[:max_sensory]
            midpoint = len(selected) // 2
            return selected, selected[:midpoint], selected[midpoint:], "fallback"

        labels = np.asarray(brain.annotations["nerve"], dtype=str)
        left = afferent[np.char.lower(np.char.strip(labels[afferent])) == "left"]
        right = afferent[np.char.lower(np.char.strip(labels[afferent])) == "right"]
        if left.size == 0 or right.size == 0:
            selected = afferent[:max_sensory]
            midpoint = len(selected) // 2
            return selected, selected[:midpoint], selected[midpoint:], "fallback"

        per_side = max(1, max_sensory // 2)
        selected_left = left[:per_side]
        selected_right = right[:per_side]
        selected = np.concatenate((selected_left, selected_right))[:max_sensory]
        return selected, selected_left, selected_right, "nerve"

    def reset(self) -> None:
        """Reset membrane potentials, refractory counters, and time."""
        n = self.model._graph.shape[0]
        self._potential = np.zeros(n, dtype=float)
        self._refractory = np.zeros(n, dtype=int)
        self._previous_spikes = np.zeros(n, dtype=bool)
        self._step = 0

    @staticmethod
    def _active_bank(bank: np.ndarray, intensity: float) -> np.ndarray:
        if bank.size == 0 or intensity <= 0.0:
            return np.empty(0, dtype=int)
        count = max(1, int(np.ceil(min(1.0, intensity) * bank.size)))
        return bank[:count]

    def _sensory_spikes(self, sensor: dict[str, float | bool]) -> np.ndarray:
        spikes = np.zeros(self.model._graph.shape[0], dtype=bool)
        left = self._active_bank(self._left_input, float(sensor.get("left", 0.0)))
        right = self._active_bank(self._right_input, float(sensor.get("right", 0.0)))
        spikes[np.concatenate((left, right))] = True
        return spikes

    def step(self, sensor: dict[str, float | bool]) -> MotorCommand:
        """Process one sensory frame and return decoded motor output."""
        sensory_spikes = self._sensory_spikes(sensor)
        current_spikes = sensory_spikes | self._previous_spikes
        input_current = self.weight_scale * np.asarray(
            current_spikes @ self.model._graph
        ).reshape(-1)
        self._potential = self._potential * (1.0 - self.leak) + input_current
        can_fire = self._refractory == 0
        spikes = can_fire & (self._potential >= self.threshold)
        self._potential[spikes] = 0.0
        self._refractory[spikes] = self.refractory_steps
        self._refractory[~spikes] = np.maximum(0, self._refractory[~spikes] - 1)
        self._previous_spikes = spikes

        behavior = self._decode(spikes)
        self._step += 1
        return MotorCommand(
            action=behavior.action,
            locomotion_drive=behavior.locomotion_drive,
            left_drive=behavior.left_drive,
            right_drive=behavior.right_drive,
            endocrine_drive=behavior.endocrine_drive,
        )

    def _decode(self, active: np.ndarray) -> BehaviorState:
        def drive(local_ids: np.ndarray) -> float:
            if local_ids.size == 0:
                return 0.0
            return float(active[local_ids].mean())

        return BehaviorState(
            step=self._step,
            locomotion_drive=drive(self.model._descending_local),
            left_drive=drive(self.model._left_local),
            right_drive=drive(self.model._right_local),
            endocrine_drive=drive(self.model._endocrine_local),
        )