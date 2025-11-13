"""Transformation tools for CAD objects"""
from typing import Tuple, Optional
import math


class TransformTool:
    """Tool for transforming CAD objects (move, rotate, scale, copy)"""

    def __init__(self, document):
        self.document = document

    def move(self, dx: float, dy: float, object_ids: Optional[list] = None):
        """Move objects by delta x and y"""
        if object_ids is None:
            # Move selected objects
            self.document.move_selected(dx, dy)
        else:
            # Move specific objects
            for obj_id in object_ids:
                geom = self.document.get_geometry_by_id(obj_id)
                if geom:
                    geom.transform(dx, dy)

    def move_to(self, target_x: float, target_y: float, from_x: float, from_y: float):
        """Move selected objects from one point to another"""
        dx = target_x - from_x
        dy = target_y - from_y
        self.move(dx, dy)

    def rotate(self, angle_degrees: float, center_x: float = 0.0, center_y: float = 0.0,
               object_ids: Optional[list] = None):
        """Rotate objects around a center point"""
        angle_rad = math.radians(angle_degrees)

        if object_ids is None:
            # Rotate selected objects
            self.document.rotate_selected(angle_rad, center_x, center_y)
        else:
            # Rotate specific objects
            for obj_id in object_ids:
                geom = self.document.get_geometry_by_id(obj_id)
                if geom:
                    # Translate to origin
                    geom.transform(-center_x, -center_y)
                    # Rotate
                    geom.transform(0, 0, angle=angle_rad)
                    # Translate back
                    geom.transform(center_x, center_y)

    def scale(self, scale_factor: float, center_x: float = 0.0, center_y: float = 0.0,
              object_ids: Optional[list] = None):
        """Scale objects around a center point"""
        if object_ids is None:
            object_ids = self.document.selected_objects

        for obj_id in object_ids:
            geom = self.document.get_geometry_by_id(obj_id)
            if geom:
                # Translate to origin
                geom.transform(-center_x, -center_y)
                # Scale
                geom.transform(0, 0, scale=scale_factor)
                # Translate back
                geom.transform(center_x, center_y)

    def copy(self, dx: float = 0.0, dy: float = 0.0):
        """Copy selected objects with optional offset"""
        return self.document.copy_selected(dx, dy)

    def mirror_horizontal(self, axis_y: float = 0.0):
        """Mirror selected objects horizontally across a line"""
        for geom in self.document.get_selected_geometries():
            # Translate to axis
            geom.transform(0, -axis_y)
            # Mirror (negate y)
            from cad_engine.geometry import Line, Point
            if isinstance(geom, Line):
                geom.start_y = -geom.start_y
                geom.end_y = -geom.end_y
            elif isinstance(geom, Point):
                geom.y = -geom.y
            # Translate back
            geom.transform(0, axis_y)

    def mirror_vertical(self, axis_x: float = 0.0):
        """Mirror selected objects vertically across a line"""
        for geom in self.document.get_selected_geometries():
            # Translate to axis
            geom.transform(-axis_x, 0)
            # Mirror (negate x)
            from cad_engine.geometry import Line, Point
            if isinstance(geom, Line):
                geom.start_x = -geom.start_x
                geom.end_x = -geom.end_x
            elif isinstance(geom, Point):
                geom.x = -geom.x
            # Translate back
            geom.transform(axis_x, 0)

    def array_rectangular(self, rows: int, cols: int, row_spacing: float, col_spacing: float):
        """Create rectangular array of selected objects"""
        if rows < 1 or cols < 1:
            return []

        created = []
        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    continue  # Skip original position

                dx = j * col_spacing
                dy = i * row_spacing
                copied = self.document.copy_selected(dx, dy)
                created.extend(copied)

        return created

    def array_polar(self, center_x: float, center_y: float, count: int, angle_degrees: float):
        """Create polar array of selected objects"""
        if count < 2:
            return []

        angle_step = angle_degrees / (count - 1)
        created = []

        for i in range(1, count):
            # Copy objects
            copied = self.document.copy_selected()

            # Rotate each copy
            angle = i * angle_step
            for geom in copied:
                # Translate to center
                geom.transform(-center_x, -center_y)
                # Rotate
                geom.transform(0, 0, angle=math.radians(angle))
                # Translate back
                geom.transform(center_x, center_y)

            created.extend(copied)

        return created
