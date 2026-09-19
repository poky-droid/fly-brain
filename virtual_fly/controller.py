"""Motor decoding for the V1 environment loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCommand:
    action: str
    locomotion_drive: float
    left_drive: float
    right_drive: float
    endocrine_drive: float = 0.0


class FlyController:
    """Convert sensory asymmetry into a deterministic motor command."""

    def decide(self, sensor: dict[str, float | bool]) -> MotorCommand:
        left = max(0.0, min(1.0, float(sensor.get("left", 0.0))))
        right = max(0.0, min(1.0, float(sensor.get("right", 0.0))))
        blocked = bool(sensor.get("blocked", False))

        if blocked:
            action = "turn_left"
            locomotion = max(left, right)
        elif right > left + 0.15:
            action = "turn_right"
            locomotion = max(left, right)
        elif left > right + 0.15:
            action = "turn_left"
            locomotion = max(left, right)
        elif max(left, right) > 0.05:
            action = "walk_forward"
            locomotion = max(left, right)
        else:
            action = "rest"
            locomotion = 0.0

        return MotorCommand(
            action=action,
            locomotion_drive=locomotion,
            left_drive=left,
            right_drive=right,
        )
