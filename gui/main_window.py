"""Main Window for CAD application"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QToolBar, QStatusBar, QFileDialog, QMessageBox,
                               QDockWidget, QListWidget, QPushButton, QLabel,
                               QInputDialog, QColorDialog)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QColor

from cad_engine import CADDocument
from .cad_viewport import CADViewport
from pdf_vectorizer import PDFVectorizer


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.document = CADDocument()
        self.pdf_vectorizer = PDFVectorizer()
        self.current_tool = None  # Current active tool

        self.init_ui()
        self.statusBar().showMessage("Ready")

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("PDF to CAD Vectorizer")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget with viewport
        self.viewport = CADViewport(self.document, self)
        self.setCentralWidget(self.viewport)

        # Connect viewport signals
        self.viewport.clicked.connect(self.on_viewport_clicked)

        # Create menus and toolbars
        self.create_menus()
        self.create_toolbars()
        self.create_layer_panel()

        # Status bar
        self.status_label = QLabel("")
        self.statusBar().addPermanentWidget(self.status_label)
        self.update_status()

    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_pdf_action = QAction("Open PDF...", self)
        open_pdf_action.setShortcut(QKeySequence.Open)
        open_pdf_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_pdf_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.triggered.connect(self.select_all)
        edit_menu.addAction(select_all_action)

        delete_action = QAction("Delete", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)

        edit_menu.addSeparator()

        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selected)
        edit_menu.addAction(copy_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.viewport.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.viewport.zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_extents_action = QAction("Zoom Extents", self)
        zoom_extents_action.setShortcut("F")
        zoom_extents_action.triggered.connect(self.viewport.zoom_extents)
        view_menu.addAction(zoom_extents_action)

        view_menu.addSeparator()

        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.viewport.reset_view)
        view_menu.addAction(reset_view_action)

        # Layer menu
        layer_menu = menubar.addMenu("&Layer")

        new_layer_action = QAction("New Layer...", self)
        new_layer_action.triggered.connect(self.new_layer)
        layer_menu.addAction(new_layer_action)

    def create_toolbars(self):
        """Create toolbars"""
        # File toolbar
        file_toolbar = QToolBar("File")
        file_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(file_toolbar)

        open_pdf_btn = QAction("Open PDF", self)
        open_pdf_btn.setStatusTip("Open PDF file")
        open_pdf_btn.triggered.connect(self.open_pdf)
        file_toolbar.addAction(open_pdf_btn)

        # View toolbar
        view_toolbar = QToolBar("View")
        view_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(view_toolbar)

        zoom_in_btn = QAction("Zoom +", self)
        zoom_in_btn.triggered.connect(self.viewport.zoom_in)
        view_toolbar.addAction(zoom_in_btn)

        zoom_out_btn = QAction("Zoom -", self)
        zoom_out_btn.triggered.connect(self.viewport.zoom_out)
        view_toolbar.addAction(zoom_out_btn)

        zoom_extents_btn = QAction("Zoom Extents", self)
        zoom_extents_btn.triggered.connect(self.viewport.zoom_extents)
        view_toolbar.addAction(zoom_extents_btn)

        view_toolbar.addSeparator()

        rotate_left_btn = QAction("Rotate Left", self)
        rotate_left_btn.triggered.connect(lambda: self.viewport.rotate_view(-15))
        view_toolbar.addAction(rotate_left_btn)

        rotate_right_btn = QAction("Rotate Right", self)
        rotate_right_btn.triggered.connect(lambda: self.viewport.rotate_view(15))
        view_toolbar.addAction(rotate_right_btn)

        # Edit toolbar
        edit_toolbar = QToolBar("Edit")
        edit_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(edit_toolbar)

        select_btn = QAction("Select", self)
        select_btn.setCheckable(True)
        select_btn.triggered.connect(lambda: self.set_tool("select"))
        edit_toolbar.addAction(select_btn)

        move_btn = QAction("Move", self)
        move_btn.setCheckable(True)
        move_btn.triggered.connect(lambda: self.set_tool("move"))
        edit_toolbar.addAction(move_btn)

        copy_btn = QAction("Copy", self)
        copy_btn.triggered.connect(self.copy_selected)
        edit_toolbar.addAction(copy_btn)

        delete_btn = QAction("Delete", self)
        delete_btn.triggered.connect(self.delete_selected)
        edit_toolbar.addAction(delete_btn)

        edit_toolbar.addSeparator()

        draw_line_btn = QAction("Draw Line", self)
        draw_line_btn.setCheckable(True)
        draw_line_btn.triggered.connect(lambda: self.set_tool("line"))
        edit_toolbar.addAction(draw_line_btn)

        draw_point_btn = QAction("Draw Point", self)
        draw_point_btn.setCheckable(True)
        draw_point_btn.triggered.connect(lambda: self.set_tool("point"))
        edit_toolbar.addAction(draw_point_btn)

    def create_layer_panel(self):
        """Create layer management panel"""
        dock = QDockWidget("Layers", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout()

        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.on_layer_selected)
        layout.addWidget(self.layer_list)

        # Layer buttons
        btn_layout = QHBoxLayout()

        new_layer_btn = QPushButton("New")
        new_layer_btn.clicked.connect(self.new_layer)
        btn_layout.addWidget(new_layer_btn)

        delete_layer_btn = QPushButton("Delete")
        delete_layer_btn.clicked.connect(self.delete_layer)
        btn_layout.addWidget(delete_layer_btn)

        layout.addLayout(btn_layout)

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.update_layer_list()

    def set_tool(self, tool_name: str):
        """Set active tool"""
        self.current_tool = tool_name
        self.statusBar().showMessage(f"Tool: {tool_name}")

    def on_viewport_clicked(self, x: float, y: float):
        """Handle viewport click"""
        if self.current_tool == "select":
            self.select_at_point(x, y)
        elif self.current_tool == "point":
            self.document.add_point(x, y)
            self.viewport.refresh()
            self.update_status()
        elif self.current_tool == "line":
            # Line tool would need two clicks - simplified for now
            self.statusBar().showMessage(f"Line tool at ({x:.2f}, {y:.2f})")

    def select_at_point(self, x: float, y: float):
        """Select geometry at point"""
        geometries = self.document.get_geometries_at_point(x, y, tolerance=5.0)
        if geometries:
            # Toggle selection of first found geometry
            geom = geometries[0]
            if geom.selected:
                self.document.deselect_object(geom.id)
            else:
                self.document.select_object(geom.id)
            self.viewport.refresh()
            self.update_status()

    def open_pdf(self):
        """Open and vectorize PDF file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf)"
        )

        if filename:
            self.statusBar().showMessage("Loading PDF...")
            if self.pdf_vectorizer.load_pdf(filename):
                # Convert to CAD data
                cad_data = self.pdf_vectorizer.convert_to_cad_units(scale=1.0)

                # Clear existing geometry
                self.document.clear()

                # Add lines to document
                for line_data in cad_data['lines']:
                    self.document.add_line(
                        line_data['start'][0], line_data['start'][1],
                        line_data['end'][0], line_data['end'][1],
                        color=line_data['color']
                    )

                self.viewport.zoom_extents()
                self.statusBar().showMessage(f"Loaded {len(cad_data['lines'])} lines from PDF")
                self.update_status()
            else:
                QMessageBox.warning(self, "Error", "Failed to load PDF file")
                self.statusBar().showMessage("Failed to load PDF")

    def select_all(self):
        """Select all objects"""
        self.document.select_all()
        self.viewport.refresh()
        self.update_status()

    def delete_selected(self):
        """Delete selected objects"""
        count = len(self.document.selected_objects)
        if count > 0:
            self.document.remove_selected()
            self.viewport.refresh()
            self.statusBar().showMessage(f"Deleted {count} objects")
            self.update_status()

    def copy_selected(self):
        """Copy selected objects with offset"""
        offset = 10.0
        copied = self.document.copy_selected(dx=offset, dy=offset)
        if copied:
            self.viewport.refresh()
            self.statusBar().showMessage(f"Copied {len(copied)} objects")
            self.update_status()

    def new_layer(self):
        """Create new layer"""
        name, ok = QInputDialog.getText(self, "New Layer", "Layer name:")
        if ok and name:
            if self.document.layer_manager.add_layer(name):
                self.update_layer_list()
                self.statusBar().showMessage(f"Created layer '{name}'")
            else:
                QMessageBox.warning(self, "Error", "Layer already exists")

    def delete_layer(self):
        """Delete selected layer"""
        current_item = self.layer_list.currentItem()
        if current_item:
            layer_name = current_item.text()
            if self.document.layer_manager.remove_layer(layer_name):
                self.update_layer_list()
                self.statusBar().showMessage(f"Deleted layer '{layer_name}'")
            else:
                QMessageBox.warning(self, "Error", "Cannot delete this layer")

    def on_layer_selected(self, item):
        """Handle layer selection"""
        layer_name = item.text()
        self.document.layer_manager.set_current_layer(layer_name)
        self.statusBar().showMessage(f"Current layer: {layer_name}")

    def update_layer_list(self):
        """Update layer list widget"""
        self.layer_list.clear()
        layers = self.document.layer_manager.get_all_layers()
        for layer in layers:
            self.layer_list.addItem(layer)

    def update_status(self):
        """Update status bar with document statistics"""
        stats = self.document.get_statistics()
        status_text = f"Objects: {stats['total_objects']} | Lines: {stats['lines']} | Points: {stats['points']} | Selected: {stats['selected']}"
        self.status_label.setText(status_text)

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key_Escape:
            self.document.clear_selection()
            self.viewport.refresh()
            self.current_tool = None
            self.statusBar().showMessage("Tool cancelled")
        elif event.key() == Qt.Key_F:
            self.viewport.zoom_extents()
        else:
            super().keyPressEvent(event)
