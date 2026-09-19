"""Minimal sensory environment for the V1 VirtualFly loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fly import Body2D


@dataclass(frozen=True)
class LightSource:
    x: float
    y: float
    intensity: float = 1.0
    radius: float = 25.0


@dataclass(frozen=True)
class FoodSource:
    x: float
    y: float
    quality: float = 1.0


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Environment2D:
    width: float = 100.0
    height: float = 100.0
    light: LightSource | None = None
    food: FoodSource | None = None
    obstacles: list[Obstacle] = field(default_factory=list)

    def contains_point(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def is_blocked(self, x: float, y: float) -> bool:
        return any(
            obstacle.x <= x <= obstacle.x + obstacle.width
            and obstacle.y <= y <= obstacle.y + obstacle.height
            for obstacle in self.obstacles
        )

    def observe(self, body: Body2D) -> dict[str, float | bool]:
        """Return heading-relative left/right light sensor values."""
        if self.light is None:
            left = right = 0.0
            light_distance = math.inf
        else:
            dx = self.light.x - body.x
            dy = self.light.y - body.y
            light_distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx) - body.heading
            falloff = max(0.0, 1.0 - light_distance / max(self.light.radius, 1e-9))
            strength = max(0.0, min(1.0, self.light.intensity * falloff))
            # Overlapping 90-degree receptive fields: a front stimulus reaches
            # both sensors, while lateral stimuli produce an asymmetric pair.
            left = strength * max(0.0, math.cos(angle - math.pi / 4.0))
            right = strength * max(0.0, math.cos(angle + math.pi / 4.0))

        food_distance = (
            math.inf
            if self.food is None
            else math.hypot(self.food.x - body.x, self.food.y - body.y)
        )
        return {
            "left": float(left),
            "right": float(right),
            "light_distance": float(light_distance),
            "food_distance": float(food_distance),
            "blocked": self.is_blocked(body.x, body.y),
        }
