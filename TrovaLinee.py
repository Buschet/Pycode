# ============================================================================
# PARAMETRI DA MODIFICARE
# ============================================================================

# Distanza massima di ricerca (le linee oltre questa distanza non vengono considerate)
MAX_DISTANCE = 100.0  # Impostare a None per nessun limite

# Numero massimo di linee da trovare per ogni punto
MAX_LINES_PER_POINT = 1  # Impostare a None per trovare tutte le linee

# Soglia di tolleranza per considerare un punto "su" una linea
TOLERANCE = 0.01  # Punti più vicini di questa distanza sono considerati sulla linea

# Selezionare automaticamente le linee trovate?
AUTO_SELECT_LINES = True  # True o False

# Mostrare informazioni dettagliate?
VERBOSE = True  # True o False

# ============================================================================




"""
Script per TROVARE LINEE (EDGE) più vicine a punti selezionati
Funzionalità INVERSA rispetto a Dividi.py

ISTRUZIONI:
1. Seleziona PUNTI (vertici) nella vista 3D
2. Modifica i parametri nella sezione PARAMETRI qui sopra
3. Esegui lo script
4. Lo script troverà le linee più vicine ai punti selezionati
"""

from PyMpc import *
import math

####### DEFAULT ##########
App.clearTerminal()
doc = App.caeDocument()


def calculate_distance(p1, p2):
	"""Calcola la distanza euclidea tra due punti"""
	dx = p2.x - p1.x
	dy = p2.y - p1.y
	dz = p2.z - p1.z
	return math.sqrt(dx*dx + dy*dy + dz*dz)


def point_to_line_distance(point, line_start, line_end):
	"""
	Calcola la distanza minima tra un punto e una linea (segmento) in 3D
	Restituisce: (distanza, punto_più_vicino_sulla_linea, parametro_t)
	"""
	# Vettore della linea
	lx = line_end.x - line_start.x
	ly = line_end.y - line_start.y
	lz = line_end.z - line_start.z

	# Lunghezza al quadrato della linea
	line_length_sq = lx*lx + ly*ly + lz*lz

	# Se la linea è praticamente un punto
	if line_length_sq < 1e-10:
		dist = calculate_distance(point, line_start)
		return dist, line_start, 0.0

	# Vettore dal punto iniziale della linea al punto
	px = point.x - line_start.x
	py = point.y - line_start.y
	pz = point.z - line_start.z

	# Proiezione del punto sulla linea (parametro t)
	t = (px*lx + py*ly + pz*lz) / line_length_sq

	# Limita t tra 0 e 1 (per rimanere sul segmento)
	t = max(0.0, min(1.0, t))

	# Punto più vicino sulla linea
	closest_x = line_start.x + t * lx
	closest_y = line_start.y + t * ly
	closest_z = line_start.z + t * lz

	# Calcola la distanza
	dx = point.x - closest_x
	dy = point.y - closest_y
	dz = point.z - closest_z
	distance = math.sqrt(dx*dx + dy*dy + dz*dz)

	# Crea un oggetto punto per il punto più vicino
	class ClosestPoint:
		def __init__(self, x, y, z):
			self.x = x
			self.y = y
			self.z = z

	closest_point = ClosestPoint(closest_x, closest_y, closest_z)

	return distance, closest_point, t


def get_all_edges_from_geometries():
	"""
	Estrae tutti gli edge da tutte le geometrie nel documento
	Restituisce una lista di dizionari con info sugli edge
	"""
	edges_list = []

	# Itera su tutte le geometrie nel documento
	for geom_id in doc.geometries.keys():
		geom = doc.geometries[geom_id]
		shape = geom.shape

		# Se è un edge singolo
		if shape.shapeType == TopAbs_ShapeEnum.TopAbs_EDGE:
			num_vertex = shape.getNumberOfSubshapes(MpcSubshapeType.Vertex)
			if num_vertex == 2:
				p1 = shape.vertexPosition(0)
				p2 = shape.vertexPosition(1)
				length = calculate_distance(p1, p2)
				edges_list.append({
					'geom': geom,
					'geom_name': geom.name,
					'geom_id': geom_id,
					'edge_id': -1,
					'p1': p1,
					'p2': p2,
					'length': length
				})

		# Se è una geometria complessa, estrai tutti i suoi edge
		else:
			num_edges = shape.getNumberOfSubshapes(MpcSubshapeType.Edge)
			for edge_id in range(num_edges):
				# Ottieni i vertici dell'edge
				vertices_ids = shape.getSubshapeChildren(
					edge_id, MpcSubshapeType.Edge, MpcSubshapeType.Vertex)

				if len(vertices_ids) == 2:
					p1 = shape.vertexPosition(vertices_ids[0])
					p2 = shape.vertexPosition(vertices_ids[1])
					length = calculate_distance(p1, p2)
					edges_list.append({
						'geom': geom,
						'geom_name': geom.name,
						'geom_id': geom_id,
						'edge_id': edge_id,
						'p1': p1,
						'p2': p2,
						'length': length
					})

	return edges_list


def get_selected_points():
	"""
	Estrae tutti i punti (vertici) dalla selezione corrente
	Restituisce una lista di dizionari con info sui punti
	"""
	points_list = []

	# Ottieni la selezione corrente
	sg = doc.scene.selectedGeometries

	if len(sg) == 0:
		return points_list

	# Itera sulle geometrie selezionate
	for geom, subselection in sg.items():
		shape = geom.shape

		# Se è un vertice singolo selezionato
		if shape.shapeType == TopAbs_ShapeEnum.TopAbs_VERTEX:
			pos = shape.vertexPosition(0)
			points_list.append({
				'geom': geom,
				'geom_name': geom.name,
				'geom_id': geom.id,
				'vertex_id': -1,
				'position': pos
			})

		# Altrimenti, cerca vertici selezionati come subshape
		else:
			for vertex_id in subselection.vertices:
				pos = shape.vertexPosition(vertex_id)
				points_list.append({
					'geom': geom,
					'geom_name': geom.name,
					'geom_id': geom.id,
					'vertex_id': vertex_id,
					'position': pos
				})

	return points_list


def find_closest_lines():
	"""
	Funzione principale: trova le linee più vicine ai punti selezionati
	"""

	print("\n" + "="*70)
	print("RICERCA LINEE PIÙ VICINE AI PUNTI SELEZIONATI")
	print("="*70)

	# Ottieni i punti selezionati
	points = get_selected_points()

	if len(points) == 0:
		print("\nERRORE: Nessun punto selezionato!")
		print("Seleziona uno o più punti (vertici) prima di eseguire lo script.")
		return

	print(f"\nPunti selezionati: {len(points)}")

	# Ottieni tutti gli edge dal documento
	print("Ricerca di tutti gli edge nel documento...")
	all_edges = get_all_edges_from_geometries()

	if len(all_edges) == 0:
		print("\nERRORE: Nessun edge trovato nel documento!")
		return

	print(f"Edge totali trovati: {len(all_edges)}")

	# Parametri
	print(f"\nParametri:")
	print(f"  Distanza massima: {MAX_DISTANCE if MAX_DISTANCE else 'Illimitata'}")
	print(f"  Max linee per punto: {MAX_LINES_PER_POINT if MAX_LINES_PER_POINT else 'Tutte'}")
	print(f"  Tolleranza: {TOLERANCE}")
	print("-"*70)

	# Per ogni punto, trova le linee più vicine
	results = []
	edges_to_select = set()  # Set di (geom, edge_id) da selezionare

	for idx, point_data in enumerate(points, 1):
		point = point_data['position']
		point_name = point_data['geom_name']

		if VERBOSE:
			print(f"\nPunto {idx}/{len(points)}: {point_name}")
			print(f"  Posizione: ({point.x:.3f}, {point.y:.3f}, {point.z:.3f})")

		# Calcola distanza da tutti gli edge
		distances = []

		for edge_data in all_edges:
			# Salta se è lo stesso oggetto del punto (per evitare auto-selezione)
			# if edge_data['geom_id'] == point_data['geom_id']:
			#     continue

			distance, closest_pt, t = point_to_line_distance(
				point, edge_data['p1'], edge_data['p2'])

			# Filtra per distanza massima se impostata
			if MAX_DISTANCE and distance > MAX_DISTANCE:
				continue

			distances.append({
				'edge': edge_data,
				'distance': distance,
				'closest_point': closest_pt,
				't': t
			})

		# Ordina per distanza
		distances.sort(key=lambda x: x['distance'])

		# Prendi solo le prime N linee se impostato
		if MAX_LINES_PER_POINT:
			distances = distances[:MAX_LINES_PER_POINT]

		if len(distances) == 0:
			if VERBOSE:
				print(f"  Nessuna linea trovata entro la distanza massima")
			continue

		if VERBOSE:
			print(f"  Linee trovate: {len(distances)}")

		# Mostra risultati
		for i, result in enumerate(distances, 1):
			edge = result['edge']
			dist = result['distance']
			t = result['t']

			# Determina se il punto è "sulla" linea
			on_line = dist < TOLERANCE

			if VERBOSE:
				status = "SULLA LINEA" if on_line else ""
				print(f"    {i}. Edge: {edge['geom_name']}", end="")
				if edge['edge_id'] >= 0:
					print(f" (Edge ID: {edge['edge_id']})", end="")
				print(f" - Distanza: {dist:.4f} {status}")
				print(f"       Parametro t: {t:.3f} (0=inizio, 1=fine)")

			# Aggiungi ai risultati
			results.append({
				'point': point_data,
				'edge': edge,
				'distance': dist,
				't': t,
				'on_line': on_line
			})

			# Aggiungi agli edge da selezionare
			edges_to_select.add((edge['geom'], edge['edge_id']))

	# Seleziona automaticamente le linee trovate
	if AUTO_SELECT_LINES and len(edges_to_select) > 0:
		print(f"\n{'='*70}")
		print("SELEZIONE AUTOMATICA DELLE LINEE TROVATE")
		print(f"{'='*70}")

		# Deseleziona tutto
		doc.scene.unselectAll()

		# Seleziona gli edge trovati
		for geom, edge_id in edges_to_select:
			if edge_id == -1:
				# È un edge singolo, seleziona l'intera geometria
				doc.scene.select(geom)
			else:
				# È un subedge, seleziona specificamente quell'edge
				doc.scene.selectSubshape(geom, edge_id, MpcSubshapeType.Edge)

		print(f"Selezionate {len(edges_to_select)} linee")

	# Riepilogo finale
	print(f"\n{'='*70}")
	print("RIEPILOGO")
	print(f"{'='*70}")
	print(f"Punti analizzati: {len(points)}")
	print(f"Linee totali nel modello: {len(all_edges)}")
	print(f"Corrispondenze trovate: {len(results)}")

	# Statistiche
	if len(results) > 0:
		min_dist = min(r['distance'] for r in results)
		max_dist = max(r['distance'] for r in results)
		avg_dist = sum(r['distance'] for r in results) / len(results)
		on_line_count = sum(1 for r in results if r['on_line'])

		print(f"\nStatistiche distanze:")
		print(f"  Minima: {min_dist:.4f}")
		print(f"  Massima: {max_dist:.4f}")
		print(f"  Media: {avg_dist:.4f}")
		print(f"  Punti sulla linea (entro tolleranza): {on_line_count}")

	print("="*70 + "\n")


# Esegui lo script
if __name__ == "__main__":
	try:
		find_closest_lines()
	except Exception as e:
		print(f"\nERRORE durante l'esecuzione: {str(e)}")
		import traceback
		traceback.print_exc()
