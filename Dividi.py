# ============================================================================
# PARAMETRI DA MODIFICARE
# ============================================================================

# Scegli la modalità: 'spacing' o 'divisions'
MODE = 'divisions'

# Se MODE = 'spacing': imposta la spaziatura tra i punti
SPACING = 10.0  # Distanza tra i punti consecutivi

# Se MODE = 'divisions': imposta il numero di divisioni
NUM_DIVISIONS = 5  # L'edge sarà diviso in questo numero di parti

# Includi i vertici iniziali e finali degli edge?
INCLUDE_ENDPOINTS = False  # True o False

# ============================================================================
















"""
Script SEMPLICE per creare punti lungo edge selezionati
Lavora con le SUBGEOMETRIE (edge specifici) selezionati

ISTRUZIONI:
1. Seleziona edge SPECIFICI nella vista 3D (non l'intera geometria)
2. Modifica i parametri nella sezione PARAMETRI qui sotto
3. Esegui lo script
"""

from PyMpc import *
import math

####### DEFALUT ##########
App.clearTerminal()
doc = App.caeDocument()


def add_geom(geom):
	"""Aggiunge geometria al documento"""
	doc.addGeometry(geom)
	doc.commitChanges()
	doc.dirty = True
	App.updateActiveView()
	App.processEvents()


def calculate_distance(p1, p2):
	"""Calcola la distanza tra due punti"""
	dx = p2.x - p1.x
	dy = p2.y - p1.y
	dz = p2.z - p1.z
	return math.sqrt(dx*dx + dy*dy + dz*dz)


def create_points_by_spacing(p1, p2, spacing):
	"""Crea punti lungo un edge con spaziatura fissa"""
	length = calculate_distance(p1, p2)
	
	if spacing <= 0 or spacing >= length:
		print(f"  ATTENZIONE: Spaziatura non valida per edge di lunghezza {length:.2f}")
		return []
	
	num_divisions = int(length / spacing)
	
	points = []
	for i in range(1, num_divisions):
		t = i * spacing / length
		x = p1.x + t * (p2.x - p1.x)
		y = p1.y + t * (p2.y - p1.y)
		z = p1.z + t * (p2.z - p1.z)
		points.append([x, y, z])
	
	return points


def create_points_by_divisions(p1, p2, num_divisions):
	"""Crea punti lungo un edge dividendolo in parti uguali"""
	if num_divisions <= 1:
		print(f"  ATTENZIONE: Numero di divisioni deve essere > 1")
		return []
	
	points = []
	for i in range(1, num_divisions):
		t = i / num_divisions
		x = p1.x + t * (p2.x - p1.x)
		y = p1.y + t * (p2.y - p1.y)
		z = p1.z + t * (p2.z - p1.z)
		points.append([x, y, z])
	
	return points


def process_edges():
	"""Processa gli edge selezionati e crea i punti"""
	
	# Ottieni la selezione corrente
	sg = doc.scene.selectedGeometries
	
	if len(sg) == 0:
		print("\nERRORE: Nessuna geometria selezionata!")
		print("Seleziona edge specifici (subshape) prima di eseguire lo script.")
		return
	
	print("\n" + "="*70)
	print("CREAZIONE PUNTI LUNGO EDGE (CON SUBGEOMETRIE)")
	print("="*70)
	
	# Raccogli tutti gli edge
	edges_data = []
	
	# Itera sulle geometrie selezionate e le loro subgeometrie
	for geom, subselection in sg.items():
		shape = geom.shape
		
		# Se è un edge intero selezionato (geometria di tipo EDGE)
		if shape.shapeType == TopAbs_ShapeEnum.TopAbs_EDGE:
			num_vertex = shape.getNumberOfSubshapes(MpcSubshapeType.Vertex)
			if num_vertex == 2:
				p1 = shape.vertexPosition(0)
				p2 = shape.vertexPosition(1)
				length = calculate_distance(p1, p2)
				edges_data.append({
					'geom_name': geom.name,
					'edge_id': -1,
					'p1': p1,
					'p2': p2,
					'length': length
				})
		
		# Altrimenti, cerca edge specifici selezionati come subshape
		else:
			# Accedi agli edge selezionati come subshape
			for edge_id in subselection.edges:
				# Ottieni i vertici dell'edge
				vertices_ids = shape.getSubshapeChildren(
					edge_id, MpcSubshapeType.Edge, MpcSubshapeType.Vertex)
				
				if len(vertices_ids) == 2:
					p1 = shape.vertexPosition(vertices_ids[0])
					p2 = shape.vertexPosition(vertices_ids[1])
					length = calculate_distance(p1, p2)
					edges_data.append({
						'geom_name': geom.name,
						'edge_id': edge_id,
						'p1': p1,
						'p2': p2,
						'length': length
					})
	
	if len(edges_data) == 0:
		print("\nERRORE: Nessun edge trovato nella selezione!")
		print("Assicurati di selezionare edge specifici (subshape).")
		return
	
	print(f"\nEdge trovati: {len(edges_data)}")
	print(f"Modalità: {MODE}")
	if MODE == 'spacing':
		print(f"Spaziatura: {SPACING}")
	else:
		print(f"Numero divisioni: {NUM_DIVISIONS}")
	print(f"Includi endpoint: {INCLUDE_ENDPOINTS}")
	print("-"*70)
	
	# Deseleziona tutto
	doc.scene.unselectAll()
	
	total_points = 0
	
	# Processa ogni edge
	for idx, edge_data in enumerate(edges_data, 1):
		p1 = edge_data['p1']
		p2 = edge_data['p2']
		length = edge_data['length']
		geom_name = edge_data['geom_name']
		edge_id = edge_data['edge_id']
		
		print(f"\nEdge {idx}/{len(edges_data)}: {geom_name}")
		if edge_id >= 0:
			print(f"  Edge ID: {edge_id}")
		print(f"  Lunghezza: {length:.3f}")
		
		# Crea i punti intermedi
		if MODE == 'spacing':
			points = create_points_by_spacing(p1, p2, SPACING)
		elif MODE == 'divisions':
			points = create_points_by_divisions(p1, p2, NUM_DIVISIONS)
		else:
			print(f"  ERRORE: Modalità '{MODE}' non valida!")
			continue
		
		# Aggiungi vertici iniziali e finali se richiesto
		if INCLUDE_ENDPOINTS:
			points.insert(0, [p1.x, p1.y, p1.z])
			points.append([p2.x, p2.y, p2.z])
		
		print(f"  Punti da creare: {len(points)}")
		
		# Crea i vertici nel documento
		for point in points:
			vertex = FxOccBuilder.makeVertex(point[0], point[1], point[2])
			geom_id = doc.geometries.getlastkey(0) + 1
			geom = MpcGeometry(geom_id, f"Point_{geom_id}", vertex)
			add_geom(geom)
			total_points += 1
	
	# Rigenera la vista
	App.runCommand('Regenerate')
	App.clearTerminal()
	
	print("\n" + "="*70)
	print(f"COMPLETATO!")
	print(f"Punti totali creati: {total_points}")
	print(f"Edge processati: {len(edges_data)}")
	print("="*70 + "\n")


# Esegui lo script
if __name__ == "__main__":
	try:
		process_edges()
	except Exception as e:
		print(f"\nERRORE durante l'esecuzione: {str(e)}")
		import traceback
		traceback.print_exc()