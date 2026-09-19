"""Simple two-dimensional fly body dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Body2D:
    """Pose and movement state for a fly in a 2D arena."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    velocity: float = 1.0
    turn_rate: float = 0.35

    def apply(self, action: str) -> None:
        """Apply one discrete motor action to the body."""
        if action == "turn_left":
            self.heading += self.turn_rate
        elif action == "turn_right":
            self.heading -= self.turn_rate
        elif action == "walk_forward":
            self.x += self.velocity * math.cos(self.heading)
            self.y += self.velocity * math.sin(self.heading)
        elif action in {"rest", "idle"}:
            pass
        else:
            raise ValueError(f"Unsupported action: {action!r}")

        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))

    def advance(self) -> None:
        """Move one velocity step along the current heading."""
        self.x += self.velocity * math.cos(self.heading)
        self.y += self.velocity * math.sin(self.heading)
