"""CAD Geometry classes"""
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class GeometryObject:
    """Base class for all geometry objects"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    layer: str = "0"
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    selected: bool = False
    visible: bool = True
    line_width: float = 1.0

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Return bounding box (min_x, min_y, max_x, max_y)"""
        raise NotImplementedError

    def transform(self, dx: float, dy: float, angle: float = 0.0, scale: float = 1.0):
        """Apply transformation to geometry"""
        raise NotImplementedError

    def contains_point(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        """Check if point is close to this geometry"""
        raise NotImplementedError

    def copy(self) -> 'GeometryObject':
        """Create a copy of this geometry"""
        raise NotImplementedError


@dataclass
class Point(GeometryObject):
    """Point geometry"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x, self.y)

    def transform(self, dx: float, dy: float, angle: float = 0.0, scale: float = 1.0):
        import math
        if angle != 0.0:
            # Rotate around origin
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            new_x = self.x * cos_a - self.y * sin_a
            new_y = self.x * sin_a + self.y * cos_a
            self.x = new_x
            self.y = new_y

        if scale != 1.0:
            self.x *= scale
            self.y *= scale
            self.z *= scale

        self.x += dx
        self.y += dy

    def contains_point(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        distance = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        return distance <= tolerance

    def copy(self) -> 'Point':
        return Point(
            x=self.x, y=self.y, z=self.z,
            layer=self.layer, color=self.color,
            line_width=self.line_width
        )


@dataclass
class Line(GeometryObject):
    """Line geometry"""
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    end_z: float = 0.0

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        min_x = min(self.start_x, self.end_x)
        min_y = min(self.start_y, self.end_y)
        max_x = max(self.start_x, self.end_x)
        max_y = max(self.start_y, self.end_y)
        return (min_x, min_y, max_x, max_y)

    def transform(self, dx: float, dy: float, angle: float = 0.0, scale: float = 1.0):
        import math
        if angle != 0.0:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Rotate start point
            new_start_x = self.start_x * cos_a - self.start_y * sin_a
            new_start_y = self.start_x * sin_a + self.start_y * cos_a
            self.start_x = new_start_x
            self.start_y = new_start_y

            # Rotate end point
            new_end_x = self.end_x * cos_a - self.end_y * sin_a
            new_end_y = self.end_x * sin_a + self.end_y * cos_a
            self.end_x = new_end_x
            self.end_y = new_end_y

        if scale != 1.0:
            self.start_x *= scale
            self.start_y *= scale
            self.start_z *= scale
            self.end_x *= scale
            self.end_y *= scale
            self.end_z *= scale

        # Translate
        self.start_x += dx
        self.start_y += dy
        self.end_x += dx
        self.end_y += dy

    def contains_point(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        """Check if point is close to the line using distance from point to line segment"""
        # Vector from start to end
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y

        if dx == 0 and dy == 0:
            # Line is actually a point
            distance = ((self.start_x - x) ** 2 + (self.start_y - y) ** 2) ** 0.5
            return distance <= tolerance

        # Parameter t of closest point on line segment
        t = max(0, min(1, ((x - self.start_x) * dx + (y - self.start_y) * dy) / (dx * dx + dy * dy)))

        # Closest point on line segment
        closest_x = self.start_x + t * dx
        closest_y = self.start_y + t * dy

        # Distance from point to closest point
        distance = ((closest_x - x) ** 2 + (closest_y - y) ** 2) ** 0.5
        return distance <= tolerance

    def copy(self) -> 'Line':
        return Line(
            start_x=self.start_x, start_y=self.start_y, start_z=self.start_z,
            end_x=self.end_x, end_y=self.end_y, end_z=self.end_z,
            layer=self.layer, color=self.color,
            line_width=self.line_width
        )

    def length(self) -> float:
        """Calculate line length"""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        dz = self.end_z - self.start_z
        return (dx*dx + dy*dy + dz*dz) ** 0.5
