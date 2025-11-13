#!/usr/bin/env python3
"""
PDF to CAD Vectorizer
Main application entry point
"""
import sys
from PySide6.QtWidgets import QApplication
from gui import MainWindow


def main():
    """Main application function"""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("PDF to CAD Vectorizer")
    app.setOrganizationName("CAD Tools")
    app.setApplicationVersion("1.0.0")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
