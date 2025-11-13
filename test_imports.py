"""Test script to verify all modules can be imported"""
import sys

def test_imports():
    """Test all module imports"""
    errors = []

    # Test pdf_vectorizer
    try:
        from pdf_vectorizer import PDFVectorizer
        print("✓ pdf_vectorizer module imported successfully")
    except Exception as e:
        errors.append(f"✗ pdf_vectorizer: {e}")

    # Test cad_engine
    try:
        from cad_engine import CADDocument, LayerManager, GeometryObject, Point, Line
        print("✓ cad_engine module imported successfully")
    except Exception as e:
        errors.append(f"✗ cad_engine: {e}")

    # Test gui
    try:
        from gui import MainWindow
        print("✓ gui module imported successfully")
    except Exception as e:
        errors.append(f"✗ gui: {e}")

    # Test tools
    try:
        from tools import SelectionTool, TransformTool, DrawingTool
        print("✓ tools module imported successfully")
    except Exception as e:
        errors.append(f"✗ tools: {e}")

    # Test basic functionality
    try:
        from cad_engine import CADDocument, Point, Line
        doc = CADDocument()

        # Test adding geometries
        point = doc.add_point(10, 20)
        line = doc.add_line(0, 0, 100, 100)

        # Test layer management
        doc.layer_manager.add_layer("Layer1")

        # Test statistics
        stats = doc.get_statistics()

        print(f"✓ Basic functionality test passed: {stats}")
    except Exception as e:
        errors.append(f"✗ Basic functionality: {e}")

    # Report results
    print("\n" + "="*50)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("ALL TESTS PASSED!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
