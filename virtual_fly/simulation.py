"""Closed-loop environment -> controller -> body simulation."""

from __future__ import annotations

from dataclasses import dataclass

from .controller import FlyController, MotorCommand
from .environment import Environment2D
from .fly import Body2D


@dataclass(frozen=True)
class SimulationStep:
    step: int
    x: float
    y: float
    heading: float
    left_sensor: float
    right_sensor: float
    locomotion_drive: float
    left_drive: float
    right_drive: float
    endocrine_drive: float
    action: str
    collision: bool = False
    requested_action: str = ""


class VirtualFlySimulator:
    """Advance a body through a sensory and motor loop."""

    def __init__(
        self,
        body: Body2D,
        environment: Environment2D,
        controller: FlyController | None = None,
        brain: object | None = None,
    ) -> None:
        self.body = body
        self.environment = environment
        self.controller = controller or FlyController()
        self.brain = brain
        self._step_index = 0

    def reset(self) -> None:
        self._step_index = 0
        if self.brain is not None and hasattr(self.brain, "reset"):
            self.brain.reset()

    def step(self) -> SimulationStep:
        sensor = self.environment.observe(self.body)
        command = self._motor_command(sensor)
        requested_action = command.action
        old_pose = (self.body.x, self.body.y, self.body.heading)
        self.body.apply(command.action)
        if command.action in {"turn_left", "turn_right"} and command.locomotion_drive > 0.05:
            self.body.advance()

        collision = (
            not self.environment.contains_point(self.body.x, self.body.y)
            or self.environment.is_blocked(self.body.x, self.body.y)
        )
        if collision:
            self.body.x, self.body.y, self.body.heading = old_pose
            command = MotorCommand("rest", 0.0, command.left_drive, command.right_drive, command.endocrine_drive)

        result = SimulationStep(
            step=self._step_index,
            x=self.body.x,
            y=self.body.y,
            heading=self.body.heading,
            left_sensor=float(sensor["left"]),
            right_sensor=float(sensor["right"]),
            locomotion_drive=command.locomotion_drive,
            left_drive=command.left_drive,
            right_drive=command.right_drive,
            endocrine_drive=command.endocrine_drive,
            action=command.action,
            collision=collision,
            requested_action=requested_action,
        )
        self._step_index += 1
        return result

    def _motor_command(self, sensor: dict[str, float | bool]) -> MotorCommand:
        if self.brain is not None and hasattr(self.brain, "step"):
            result = self.brain.step(sensor)
            if isinstance(result, MotorCommand):
                return result
            if isinstance(result, dict):
                fallback = self.controller.decide(sensor)
                return MotorCommand(
                    action=str(result.get("action", fallback.action)),
                    locomotion_drive=float(result.get("locomotion_drive", fallback.locomotion_drive)),
                    left_drive=float(result.get("left_drive", fallback.left_drive)),
                    right_drive=float(result.get("right_drive", fallback.right_drive)),
                    endocrine_drive=float(result.get("endocrine_drive", fallback.endocrine_drive)),
                )
        return self.controller.decide(sensor)

    def run(self, steps: int) -> list[SimulationStep]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return [self.step() for _ in range(steps)]
