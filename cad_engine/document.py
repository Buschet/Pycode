"""CAD Document - manages all geometry and layers"""
from typing import List, Optional, Tuple
from .geometry import GeometryObject, Point, Line
from .layer_manager import LayerManager


class CADDocument:
    """Main CAD document that holds all geometry"""

    def __init__(self):
        self.geometries: List[GeometryObject] = []
        self.layer_manager = LayerManager()
        self.selected_objects: List[str] = []  # List of selected object IDs
        self.modified = False

    def add_geometry(self, geometry: GeometryObject):
        """Add geometry to document"""
        self.geometries.append(geometry)
        self.modified = True

    def remove_geometry(self, geometry_id: str) -> bool:
        """Remove geometry by ID"""
        for i, geom in enumerate(self.geometries):
            if geom.id == geometry_id:
                self.geometries.pop(i)
                if geometry_id in self.selected_objects:
                    self.selected_objects.remove(geometry_id)
                self.modified = True
                return True
        return False

    def remove_selected(self):
        """Remove all selected geometries"""
        ids_to_remove = self.selected_objects.copy()
        for geom_id in ids_to_remove:
            self.remove_geometry(geom_id)

    def get_geometry_by_id(self, geometry_id: str) -> Optional[GeometryObject]:
        """Get geometry by ID"""
        for geom in self.geometries:
            if geom.id == geometry_id:
                return geom
        return None

    def get_geometries_at_point(self, x: float, y: float, tolerance: float = 5.0) -> List[GeometryObject]:
        """Find geometries near a point"""
        result = []
        for geom in self.geometries:
            if geom.visible and geom.contains_point(x, y, tolerance):
                result.append(geom)
        return result

    def get_geometries_in_layer(self, layer: str) -> List[GeometryObject]:
        """Get all geometries in a specific layer"""
        return [geom for geom in self.geometries if geom.layer == layer]

    def get_visible_geometries(self) -> List[GeometryObject]:
        """Get all visible geometries"""
        result = []
        for geom in self.geometries:
            if geom.visible and self.layer_manager.is_layer_visible(geom.layer):
                result.append(geom)
        return result

    def select_object(self, geometry_id: str):
        """Select an object"""
        geom = self.get_geometry_by_id(geometry_id)
        if geom and geometry_id not in self.selected_objects:
            self.selected_objects.append(geometry_id)
            geom.selected = True

    def deselect_object(self, geometry_id: str):
        """Deselect an object"""
        if geometry_id in self.selected_objects:
            self.selected_objects.remove(geometry_id)
            geom = self.get_geometry_by_id(geometry_id)
            if geom:
                geom.selected = False

    def select_all(self):
        """Select all visible objects"""
        self.clear_selection()
        for geom in self.get_visible_geometries():
            self.select_object(geom.id)

    def clear_selection(self):
        """Clear all selections"""
        for geom_id in self.selected_objects.copy():
            self.deselect_object(geom_id)

    def get_selected_geometries(self) -> List[GeometryObject]:
        """Get all selected geometries"""
        return [self.get_geometry_by_id(gid) for gid in self.selected_objects if self.get_geometry_by_id(gid)]

    def move_selected(self, dx: float, dy: float):
        """Move all selected objects"""
        for geom in self.get_selected_geometries():
            geom.transform(dx, dy)
        self.modified = True

    def rotate_selected(self, angle: float, center_x: float = 0.0, center_y: float = 0.0):
        """Rotate selected objects around a point"""
        for geom in self.get_selected_geometries():
            # Translate to origin
            geom.transform(-center_x, -center_y)
            # Rotate
            geom.transform(0, 0, angle=angle)
            # Translate back
            geom.transform(center_x, center_y)
        self.modified = True

    def copy_selected(self, dx: float = 0.0, dy: float = 0.0) -> List[GeometryObject]:
        """Copy selected objects with optional offset"""
        copied = []
        for geom in self.get_selected_geometries():
            new_geom = geom.copy()
            if dx != 0.0 or dy != 0.0:
                new_geom.transform(dx, dy)
            self.add_geometry(new_geom)
            copied.append(new_geom)
        return copied

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of all visible geometry"""
        visible_geoms = self.get_visible_geometries()
        if not visible_geoms:
            return None

        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for geom in visible_geoms:
            bbox = geom.get_bounding_box()
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

        return (min_x, min_y, max_x, max_y)

    def add_line(self, x1: float, y1: float, x2: float, y2: float,
                 layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Convenience method to add a line"""
        if layer is None:
            layer = self.layer_manager.get_current_layer()
        if color is None:
            layer_obj = self.layer_manager.get_layer(layer)
            color = layer_obj.color if layer_obj else (1.0, 1.0, 1.0)

        line = Line(
            start_x=x1, start_y=y1, start_z=0.0,
            end_x=x2, end_y=y2, end_z=0.0,
            layer=layer, color=color
        )
        self.add_geometry(line)
        return line

    def add_point(self, x: float, y: float,
                  layer: Optional[str] = None, color: Optional[Tuple[float, float, float]] = None):
        """Convenience method to add a point"""
        if layer is None:
            layer = self.layer_manager.get_current_layer()
        if color is None:
            layer_obj = self.layer_manager.get_layer(layer)
            color = layer_obj.color if layer_obj else (1.0, 1.0, 1.0)

        point = Point(
            x=x, y=y, z=0.0,
            layer=layer, color=color
        )
        self.add_geometry(point)
        return point

    def clear(self):
        """Clear all geometry"""
        self.geometries.clear()
        self.selected_objects.clear()
        self.modified = False

    def get_statistics(self) -> dict:
        """Get document statistics"""
        return {
            'total_objects': len(self.geometries),
            'lines': sum(1 for g in self.geometries if isinstance(g, Line)),
            'points': sum(1 for g in self.geometries if isinstance(g, Point)),
            'selected': len(self.selected_objects),
            'layers': len(self.layer_manager.get_all_layers())
        }
