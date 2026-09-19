import math
import unittest

import numpy as np
from scipy.sparse import csr_array

from virtual_brain import Connectome
from virtual_fly import Body2D, Environment2D, LightSource, Obstacle, VirtualFlySimulator
from virtual_fly import FlyWireBrain


class VirtualFlyV1Tests(unittest.TestCase):
    def _make_connectome(self):
        graph = csr_array(
            [
                [0, 0, 100, 0, 0, 0],
                [0, 0, 0, 100, 0, 0],
                [0, 0, 0, 0, 100, 0],
                [0, 0, 0, 0, 0, 100],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        annotations = {
            "flow": ("afferent", "afferent", "intrinsic", "intrinsic", "efferent", "efferent"),
            "super_class": ("sensory", "sensory", "", "", "descending", "descending"),
            "nerve": ("", "", "", "", "left ", "right"),
        }
        return Connectome(
            matrix=graph,
            annotations=annotations,
            coordinates=np.zeros((6, 3), dtype=float),
        )

    def test_light_to_right_produces_right_sensor_signal(self):
        environment = Environment2D(
            width=20.0,
            height=20.0,
            light=LightSource(x=8.0, y=2.0, radius=20.0),
        )
        sensor = environment.observe(Body2D(x=4.0, y=4.0))
        self.assertGreater(sensor["right"], sensor["left"])

    def test_light_ahead_reaches_both_sensors(self):
        environment = Environment2D(
            width=20.0,
            height=20.0,
            light=LightSource(x=8.0, y=4.0, radius=20.0),
        )
        sensor = environment.observe(Body2D(x=4.0, y=4.0, heading=0.0))
        self.assertGreater(sensor["left"], 0.0)
        self.assertGreater(sensor["right"], 0.0)
        self.assertAlmostEqual(sensor["left"], sensor["right"])

    def test_body_turns_and_walks(self):
        body = Body2D(velocity=1.0, turn_rate=math.pi / 2.0)
        body.apply("turn_right")
        self.assertAlmostEqual(body.heading, -math.pi / 2.0)
        body.apply("walk_forward")
        self.assertAlmostEqual(body.y, -1.0)

    def test_simulator_produces_in_bounds_trajectory(self):
        body = Body2D(x=2.0, y=2.0, heading=0.0)
        environment = Environment2D(
            width=20.0,
            height=20.0,
            light=LightSource(x=16.0, y=10.0, radius=30.0),
        )
        trajectory = VirtualFlySimulator(body, environment).run(steps=5)
        self.assertEqual(len(trajectory), 5)
        self.assertNotEqual((trajectory[0].x, trajectory[0].y), (trajectory[-1].x, trajectory[-1].y))
        for state in trajectory:
            self.assertTrue(0.0 <= state.x <= environment.width)
            self.assertTrue(0.0 <= state.y <= environment.height)

    def test_flywire_brain_propagates_sensor_to_motor(self):
        brain = FlyWireBrain(
            self._make_connectome(),
            max_sensory=2,
            hops=2,
            threshold=0.5,
            leak=0.9,
            weight_scale=0.01,
            refractory_steps=0,
        )

        first = brain.step({"left": 1.0, "right": 0.0})
        second = brain.step({"left": 1.0, "right": 0.0})

        self.assertEqual(first.action, "rest")
        self.assertGreater(second.locomotion_drive, 0.0)
        self.assertEqual(second.action, "turn_left")

    def test_flywire_brain_uses_nerve_labels_when_available(self):
        connectome = self._make_connectome()
        connectome.annotations["nerve"] = ("left", "right", "", "", "left ", "right")
        brain = FlyWireBrain(connectome, max_sensory=2, hops=2)
        self.assertEqual(brain.mapping_source, "nerve")
        np.testing.assert_array_equal(brain._left_sensory_ids, [0])
        np.testing.assert_array_equal(brain._right_sensory_ids, [1])

    def test_flywire_brain_reset_replays_same_sequence(self):
        brain = FlyWireBrain(self._make_connectome(), max_sensory=2, hops=2, weight_scale=0.01, refractory_steps=0)
        sensors = [{"left": 1.0, "right": 0.0}] * 3
        first = [brain.step(sensor).action for sensor in sensors]
        brain.reset()
        second = [brain.step(sensor).action for sensor in sensors]
        self.assertEqual(first, second)

    def test_simulator_can_drive_body_from_flywire_brain(self):
        brain = FlyWireBrain(
            self._make_connectome(),
            max_sensory=2,
            hops=2,
            weight_scale=0.01,
            refractory_steps=0,
        )
        body = Body2D(x=2.0, y=2.0, velocity=0.5)
        environment = Environment2D(
            width=20.0,
            height=20.0,
            light=LightSource(x=2.0, y=8.0, radius=20.0),
        )
        trajectory = VirtualFlySimulator(body, environment, brain=brain).run(steps=3)
        self.assertEqual(trajectory[0].action, "rest")
        self.assertTrue(any((state.x, state.y) != (2.0, 2.0) for state in trajectory[1:]))

    def test_simulator_records_obstacle_collision(self):
        body = Body2D(x=1.0, y=1.0, heading=0.0, velocity=0.5)
        environment = Environment2D(
            width=20.0,
            height=20.0,
            light=LightSource(x=15.0, y=1.0, radius=20.0),
            obstacles=[Obstacle(x=1.4, y=0.0, width=2.0, height=2.0)],
        )
        state = VirtualFlySimulator(body, environment).step()
        self.assertTrue(state.collision)
        self.assertEqual(state.action, "rest")
        self.assertEqual(state.requested_action, "walk_forward")


if __name__ == "__main__":
    unittest.main()
