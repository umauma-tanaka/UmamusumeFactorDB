"""Geometry primitives used by image processing and debug output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    @classmethod
    def from_image_shape(cls, shape: tuple[int, ...]) -> "Size":
        return cls(width=int(shape[1]), height=int(shape[0]))

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_tuple(cls, value: tuple[int, int, int, int]) -> "Rect":
        return cls(*map(int, value))

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def translate(self, dx: int = 0, dy: int = 0) -> "Rect":
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def scale(self, factor: float) -> "Rect":
        return Rect(
            int(round(self.x0 * factor)),
            int(round(self.y0 * factor)),
            int(round(self.x1 * factor)),
            int(round(self.y1 * factor)),
        )

    def clamp(self, bounds: Size) -> "Rect":
        return Rect(
            min(max(0, self.x0), bounds.width),
            min(max(0, self.y0), bounds.height),
            min(max(0, self.x1), bounds.width),
            min(max(0, self.y1), bounds.height),
        )

