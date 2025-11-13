"""Drawing tools for creating CAD geometry"""
from typing import Optional, Tuple
from cad_engine.geometry import Point, Line


class DrawingTool:
    """Tool for drawing new CAD geometry"""

    def __init__(self, document):
        self.document = document
        self.current_tool = None
        self.temp_points = []  # Temporary points for multi-click tools

    def draw_point(self, x: float, y: float, layer: Optional[str] = None,
                   color: Optional[Tuple[float, float, float]] = None):
        """Draw a point"""
        point = self.document.add_point(x, y, layer, color)
        return point

    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Draw a line"""
        line = self.document.add_line(x1, y1, x2, y2, layer, color)
        return line

    def start_line(self, x: float, y: float):
        """Start drawing a line (first click)"""
        self.current_tool = "line"
        self.temp_points = [(x, y)]

    def finish_line(self, x: float, y: float):
        """Finish drawing a line (second click)"""
        if self.current_tool == "line" and len(self.temp_points) == 1:
            x1, y1 = self.temp_points[0]
            line = self.draw_line(x1, y1, x, y)
            self.temp_points.clear()
            self.current_tool = None
            return line
        return None

    def cancel_current(self):
        """Cancel current drawing operation"""
        self.temp_points.clear()
        self.current_tool = None

    def draw_rectangle(self, x1: float, y1: float, x2: float, y2: float,
                       layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Draw a rectangle using 4 lines"""
        lines = []
        # Bottom
        lines.append(self.draw_line(x1, y1, x2, y1, layer, color))
        # Right
        lines.append(self.draw_line(x2, y1, x2, y2, layer, color))
        # Top
        lines.append(self.draw_line(x2, y2, x1, y2, layer, color))
        # Left
        lines.append(self.draw_line(x1, y2, x1, y1, layer, color))
        return lines

    def draw_circle_approx(self, center_x: float, center_y: float, radius: float, segments: int = 32,
                           layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Draw a circle approximation using line segments"""
        import math
        lines = []
        angle_step = 2 * math.pi / segments

        for i in range(segments):
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            x1 = center_x + radius * math.cos(angle1)
            y1 = center_y + radius * math.sin(angle1)
            x2 = center_x + radius * math.cos(angle2)
            y2 = center_y + radius * math.sin(angle2)

            lines.append(self.draw_line(x1, y1, x2, y2, layer, color))

        return lines

    def draw_polyline(self, points: list, closed: bool = False,
                      layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Draw a polyline from a list of points"""
        if len(points) < 2:
            return []

        lines = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            lines.append(self.draw_line(x1, y1, x2, y2, layer, color))

        # Close the polyline if requested
        if closed and len(points) > 2:
            x1, y1 = points[-1]
            x2, y2 = points[0]
            lines.append(self.draw_line(x1, y1, x2, y2, layer, color))

        return lines

    def draw_grid(self, min_x: float, min_y: float, max_x: float, max_y: float,
                  spacing: float, layer: Optional[str] = None,
                  color: Optional[Tuple[float, float, float]] = None):
        """Draw a grid"""
        lines = []

        # Vertical lines
        x = min_x
        while x <= max_x:
            lines.append(self.draw_line(x, min_y, x, max_y, layer, color))
            x += spacing

        # Horizontal lines
        y = min_y
        while y <= max_y:
            lines.append(self.draw_line(min_x, y, max_x, y, layer, color))
            y += spacing

        return lines
