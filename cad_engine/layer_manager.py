"""Layer Manager for CAD"""
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Layer:
    """Represents a CAD layer"""
    name: str
    visible: bool = True
    locked: bool = False
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    line_width: float = 1.0


class LayerManager:
    """Manages CAD layers"""

    def __init__(self):
        self.layers: Dict[str, Layer] = {}
        self.current_layer = "0"
        # Create default layer
        self.add_layer("0", color=(1.0, 1.0, 1.0))

    def add_layer(self, name: str, visible: bool = True,
                  color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  line_width: float = 1.0) -> bool:
        """Add a new layer"""
        if name in self.layers:
            return False

        self.layers[name] = Layer(
            name=name,
            visible=visible,
            color=color,
            line_width=line_width
        )
        return True

    def remove_layer(self, name: str) -> bool:
        """Remove a layer (cannot remove layer 0 or current layer)"""
        if name == "0" or name == self.current_layer:
            return False

        if name in self.layers:
            del self.layers[name]
            return True
        return False

    def set_layer_visible(self, name: str, visible: bool) -> bool:
        """Set layer visibility"""
        if name in self.layers:
            self.layers[name].visible = visible
            return True
        return False

    def set_layer_locked(self, name: str, locked: bool) -> bool:
        """Set layer locked state"""
        if name in self.layers:
            self.layers[name].locked = locked
            return True
        return False

    def set_layer_color(self, name: str, color: Tuple[float, float, float]) -> bool:
        """Set layer color"""
        if name in self.layers:
            self.layers[name].color = color
            return True
        return False

    def set_current_layer(self, name: str) -> bool:
        """Set current active layer"""
        if name in self.layers and not self.layers[name].locked:
            self.current_layer = name
            return True
        return False

    def get_current_layer(self) -> str:
        """Get current layer name"""
        return self.current_layer

    def get_layer(self, name: str) -> Layer:
        """Get layer by name"""
        return self.layers.get(name)

    def get_all_layers(self) -> List[str]:
        """Get list of all layer names"""
        return list(self.layers.keys())

    def is_layer_visible(self, name: str) -> bool:
        """Check if layer is visible"""
        if name in self.layers:
            return self.layers[name].visible
        return False

    def is_layer_locked(self, name: str) -> bool:
        """Check if layer is locked"""
        if name in self.layers:
            return self.layers[name].locked
        return False
