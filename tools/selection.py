"""Selection tool for CAD objects"""
from typing import List, Tuple, Optional
from cad_engine.geometry import GeometryObject


class SelectionTool:
    """Tool for selecting CAD objects"""

    def __init__(self, document):
        self.document = document
        self.selection_mode = "single"  # single, multiple, window

    def select_at_point(self, x: float, y: float, tolerance: float = 5.0, add_to_selection: bool = False):
        """Select objects at a point"""
        geometries = self.document.get_geometries_at_point(x, y, tolerance)

        if not add_to_selection:
            self.document.clear_selection()

        for geom in geometries:
            self.document.select_object(geom.id)

        return len(geometries) > 0

    def select_in_window(self, x1: float, y1: float, x2: float, y2: float, add_to_selection: bool = False):
        """Select all objects within a rectangular window"""
        if not add_to_selection:
            self.document.clear_selection()

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        selected_count = 0
        for geom in self.document.get_visible_geometries():
            bbox = geom.get_bounding_box()
            # Check if bounding box is completely inside window
            if (bbox[0] >= min_x and bbox[2] <= max_x and
                bbox[1] >= min_y and bbox[3] <= max_y):
                self.document.select_object(geom.id)
                selected_count += 1

        return selected_count

    def select_crossing_window(self, x1: float, y1: float, x2: float, y2: float, add_to_selection: bool = False):
        """Select all objects crossing a rectangular window"""
        if not add_to_selection:
            self.document.clear_selection()

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        selected_count = 0
        for geom in self.document.get_visible_geometries():
            bbox = geom.get_bounding_box()
            # Check if bounding box intersects with window
            if not (bbox[2] < min_x or bbox[0] > max_x or
                    bbox[3] < min_y or bbox[1] > max_y):
                self.document.select_object(geom.id)
                selected_count += 1

        return selected_count

    def toggle_selection(self, x: float, y: float, tolerance: float = 5.0):
        """Toggle selection at point"""
        geometries = self.document.get_geometries_at_point(x, y, tolerance)

        for geom in geometries:
            if geom.selected:
                self.document.deselect_object(geom.id)
            else:
                self.document.select_object(geom.id)

        return len(geometries) > 0

    def get_selection_center(self) -> Optional[Tuple[float, float]]:
        """Get center point of selected objects"""
        selected = self.document.get_selected_geometries()
        if not selected:
            return None

        sum_x = 0.0
        sum_y = 0.0
        count = 0

        for geom in selected:
            bbox = geom.get_bounding_box()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            sum_x += center_x
            sum_y += center_y
            count += 1

        if count == 0:
            return None

        return (sum_x / count, sum_y / count)
