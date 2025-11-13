"""CAD Viewport widget using OpenCascade"""
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QWheelEvent, QMouseEvent
import math
from typing import Optional, Tuple

try:
    from OCC.Display.backend import load_backend
    load_backend('qt-pyside6')
    from OCC.Display.qtDisplay import qtViewer3d
    OPENCASCADE_AVAILABLE = True
except ImportError:
    OPENCASCADE_AVAILABLE = False
    print("Warning: pythonocc-core not available, using fallback 2D renderer")


class CADViewport(QWidget):
    """CAD Viewport with pan, zoom, and rotation capabilities"""

    clicked = Signal(float, float)  # Emitted when viewport is clicked (x, y in CAD coords)
    selection_changed = Signal()

    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.document = document

        # View transformation parameters
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.rotation = 0.0  # in radians

        # Mouse interaction
        self.last_mouse_pos = None
        self.is_panning = False
        self.is_selecting = False

        # Selection
        self.selection_tolerance = 5.0

        # Try to use OpenCascade viewer if available
        self.use_occ = OPENCASCADE_AVAILABLE
        self.occ_viewer = None

        if self.use_occ:
            self._setup_occ_viewer()
        else:
            self._setup_2d_viewer()

    def _setup_occ_viewer(self):
        """Setup OpenCascade 3D viewer"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            self.occ_viewer = qtViewer3d(self)
            layout.addWidget(self.occ_viewer)
            self.setLayout(layout)
        except Exception as e:
            print(f"Error setting up OCC viewer: {e}")
            self.use_occ = False
            self._setup_2d_viewer()

    def _setup_2d_viewer(self):
        """Setup fallback 2D viewer"""
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, event):
        """Paint the CAD view (fallback 2D renderer)"""
        if self.use_occ:
            return  # OCC handles its own rendering

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Apply transformations
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self.zoom, -self.zoom)  # Flip Y axis for CAD coords
        painter.translate(self.pan_x, self.pan_y)
        painter.rotate(math.degrees(self.rotation))

        # Draw grid
        self._draw_grid(painter)

        # Draw all visible geometries
        visible_geoms = self.document.get_visible_geometries()

        for geom in visible_geoms:
            self._draw_geometry(painter, geom)

        painter.end()

    def _draw_grid(self, painter):
        """Draw background grid"""
        grid_size = 50.0
        grid_color = QColor(60, 60, 60)

        # Calculate visible range
        view_width = self.width() / self.zoom
        view_height = self.height() / self.zoom

        min_x = -view_width / 2 - self.pan_x
        max_x = view_width / 2 - self.pan_x
        min_y = -view_height / 2 - self.pan_y
        max_y = view_height / 2 - self.pan_y

        # Draw vertical lines
        pen = QPen(grid_color, 0.5 / self.zoom)
        painter.setPen(pen)

        x = int(min_x / grid_size) * grid_size
        while x <= max_x:
            painter.drawLine(int(x), int(min_y), int(x), int(max_y))
            x += grid_size

        # Draw horizontal lines
        y = int(min_y / grid_size) * grid_size
        while y <= max_y:
            painter.drawLine(int(min_x), int(y), int(max_x), int(y))
            y += grid_size

        # Draw axes
        axis_pen = QPen(QColor(100, 100, 100), 1.0 / self.zoom)
        painter.setPen(axis_pen)
        painter.drawLine(int(min_x), 0, int(max_x), 0)  # X axis
        painter.drawLine(0, int(min_y), 0, int(max_y))  # Y axis

    def _draw_geometry(self, painter, geom):
        """Draw a single geometry object"""
        from cad_engine.geometry import Line, Point

        # Determine color
        if geom.selected:
            color = QColor(255, 255, 0)  # Yellow for selected
        else:
            r, g, b = geom.color
            color = QColor(int(r * 255), int(g * 255), int(b * 255))

        pen = QPen(color, geom.line_width / self.zoom)
        painter.setPen(pen)

        if isinstance(geom, Line):
            painter.drawLine(
                int(geom.start_x), int(geom.start_y),
                int(geom.end_x), int(geom.end_y)
            )
        elif isinstance(geom, Point):
            # Draw point as small cross
            size = 3.0 / self.zoom
            painter.drawLine(
                int(geom.x - size), int(geom.y),
                int(geom.x + size), int(geom.y)
            )
            painter.drawLine(
                int(geom.x), int(geom.y - size),
                int(geom.x), int(geom.y + size)
            )

    def screen_to_cad(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to CAD coordinates"""
        # Center
        x = screen_x - self.width() / 2
        y = screen_y - self.height() / 2

        # Unzoom
        x /= self.zoom
        y /= -self.zoom  # Flip Y

        # Unpan
        x -= self.pan_x
        y -= self.pan_y

        # Unrotate (if needed, currently rotation is applied to all geometry)
        if self.rotation != 0.0:
            cos_a = math.cos(-self.rotation)
            sin_a = math.sin(-self.rotation)
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            x, y = new_x, new_y

        return (x, y)

    def cad_to_screen(self, cad_x: float, cad_y: float) -> Tuple[int, int]:
        """Convert CAD coordinates to screen coordinates"""
        x, y = cad_x, cad_y

        # Rotate
        if self.rotation != 0.0:
            cos_a = math.cos(self.rotation)
            sin_a = math.sin(self.rotation)
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            x, y = new_x, new_y

        # Pan
        x += self.pan_x
        y += self.pan_y

        # Zoom
        x *= self.zoom
        y *= -self.zoom  # Flip Y

        # Center
        x += self.width() / 2
        y += self.height() / 2

        return (int(x), int(y))

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # Convert to CAD coordinates and emit signal
            cad_x, cad_y = self.screen_to_cad(event.pos().x(), event.pos().y())
            self.clicked.emit(cad_x, cad_y)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        if self.is_panning and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x() / self.zoom
            self.pan_y -= delta.y() / self.zoom
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.zoom *= zoom_factor
        else:
            self.zoom /= zoom_factor

        self.zoom = max(0.01, min(100.0, self.zoom))
        self.update()

    def zoom_extents(self):
        """Zoom to fit all geometry"""
        bbox = self.document.get_bounding_box()
        if not bbox:
            return

        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        if width == 0 or height == 0:
            return

        # Calculate zoom to fit
        margin = 1.1  # 10% margin
        zoom_x = self.width() / (width * margin)
        zoom_y = self.height() / (height * margin)
        self.zoom = min(zoom_x, zoom_y)

        # Center view
        self.pan_x = -(min_x + max_x) / 2
        self.pan_y = -(min_y + max_y) / 2

        self.update()

    def zoom_in(self):
        """Zoom in"""
        self.zoom *= 1.2
        self.update()

    def zoom_out(self):
        """Zoom out"""
        self.zoom /= 1.2
        self.update()

    def rotate_view(self, angle_degrees: float):
        """Rotate view by angle in degrees"""
        self.rotation += math.radians(angle_degrees)
        self.update()

    def reset_view(self):
        """Reset view to default"""
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.rotation = 0.0
        self.update()

    def refresh(self):
        """Refresh the viewport"""
        self.update()
