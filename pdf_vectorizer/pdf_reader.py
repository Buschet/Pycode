"""PDF Vectorizer - Extracts vector data from PDF files"""
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple, Dict, Any


class PDFVectorizer:
    """Converts PDF content to vectorial CAD data"""

    def __init__(self):
        self.points = []
        self.lines = []
        self.page_data = []

    def load_pdf(self, pdf_path: str) -> bool:
        """Load and parse PDF file"""
        try:
            doc = fitz.open(pdf_path)
            self.page_data = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_dict = self._extract_page_vectors(page, page_num)
                self.page_data.append(page_dict)

            doc.close()
            return True
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False

    def _extract_page_vectors(self, page, page_num: int) -> Dict[str, Any]:
        """Extract vector data from a single page"""
        page_dict = {
            'page_num': page_num,
            'width': page.rect.width,
            'height': page.rect.height,
            'lines': [],
            'points': [],
            'curves': [],
            'text_positions': []
        }

        # Extract paths (lines, curves, etc.)
        paths = page.get_drawings()

        for path in paths:
            for item in path['items']:
                if item[0] == 'l':  # Line
                    # item format: ('l', Point(x1, y1), Point(x2, y2))
                    p1 = item[1]
                    p2 = item[2]
                    page_dict['lines'].append({
                        'start': (p1.x, p1.y),
                        'end': (p2.x, p2.y),
                        'color': path.get('color', (0, 0, 0)),
                        'width': path.get('width', 1.0)
                    })
                elif item[0] == 'c':  # Curve (bezier)
                    # item format: ('c', P1, P2, P3, P4) - cubic bezier
                    page_dict['curves'].append({
                        'points': [(p.x, p.y) for p in item[1:]],
                        'color': path.get('color', (0, 0, 0)),
                        'width': path.get('width', 1.0)
                    })

        # Extract text positions as reference points
        text_instances = page.get_text("dict")
        for block in text_instances.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox")
                        if bbox:
                            page_dict['text_positions'].append({
                                'position': (bbox[0], bbox[1]),
                                'text': span.get('text', '')
                            })

        return page_dict

    def get_all_lines(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get all lines from all pages"""
        all_lines = []
        for page_data in self.page_data:
            for line in page_data['lines']:
                all_lines.append((line['start'], line['end']))
        return all_lines

    def get_all_points(self) -> List[Tuple[float, float]]:
        """Get all significant points from all pages"""
        all_points = []
        for page_data in self.page_data:
            # Add line endpoints as points
            for line in page_data['lines']:
                all_points.append(line['start'])
                all_points.append(line['end'])

            # Add curve control points
            for curve in page_data['curves']:
                all_points.extend(curve['points'])

            # Add text positions
            for text_pos in page_data['text_positions']:
                all_points.append(text_pos['position'])

        return all_points

    def get_page_data(self, page_num: int = 0) -> Dict[str, Any]:
        """Get data for a specific page"""
        if 0 <= page_num < len(self.page_data):
            return self.page_data[page_num]
        return {}

    def get_page_count(self) -> int:
        """Get total number of pages"""
        return len(self.page_data)

    def convert_to_cad_units(self, scale: float = 1.0) -> Dict[str, List]:
        """Convert PDF coordinates to CAD units with optional scaling"""
        cad_data = {
            'lines': [],
            'points': [],
            'curves': []
        }

        for page_data in self.page_data:
            # Convert lines
            for line in page_data['lines']:
                cad_data['lines'].append({
                    'start': (line['start'][0] * scale, line['start'][1] * scale),
                    'end': (line['end'][0] * scale, line['end'][1] * scale),
                    'color': line['color'],
                    'width': line['width']
                })

            # Convert curves (approximate with line segments)
            for curve in page_data['curves']:
                # Simple approximation: connect control points
                points = curve['points']
                for i in range(len(points) - 1):
                    cad_data['lines'].append({
                        'start': (points[i][0] * scale, points[i][1] * scale),
                        'end': (points[i+1][0] * scale, points[i+1][1] * scale),
                        'color': curve['color'],
                        'width': curve['width']
                    })

        return cad_data
