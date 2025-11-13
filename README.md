# PDF to CAD Vectorizer

Un'applicazione CAD completa con capacità di importare e vettorializzare file PDF. Sviluppata in Python con Qt (PySide6) e OpenCascade.

## Caratteristiche

### Import PDF
- Caricamento file PDF
- Estrazione automatica di linee e curve
- Conversione in elementi CAD vettoriali

### Strumenti CAD

#### Visualizzazione
- **Zoom**: Rotella del mouse o pulsanti toolbar
- **Pan**: Pulsante centrale del mouse (drag)
- **Ruota Vista**: Rotazione della vista di 15° a sinistra/destra
- **Zoom Extents**: Adatta la vista a tutti gli oggetti (tasto F)
- **Reset Vista**: Ripristina vista predefinita

#### Editing
- **Seleziona**: Click su oggetti per selezionarli
- **Muovi**: Sposta oggetti selezionati
- **Copia**: Duplica oggetti selezionati con offset
- **Elimina**: Rimuove oggetti selezionati (tasto Delete)
- **Disegna Punto**: Crea punti
- **Disegna Linea**: Crea linee

#### Layer
- Gestione layer multipli
- Creazione/eliminazione layer
- Visibilità e blocco layer
- Assegnazione colori per layer

### Scorciatoie da Tastiera

- `Ctrl+O`: Apri PDF
- `Ctrl+A`: Seleziona tutto
- `Ctrl+C`: Copia selezione
- `Delete`: Elimina selezione
- `Esc`: Annulla operazione corrente
- `F`: Zoom extents
- `Ctrl++`: Zoom in
- `Ctrl+-`: Zoom out

## Installazione

### Prerequisiti

Assicurati di avere Python 3.8+ installato. Hai già installato:
- Python
- OpenCascade
- Qt
- cmake

### Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### Dipendenze

- **PySide6**: Framework GUI Qt per Python
- **PyMuPDF**: Lettura e analisi PDF
- **pythonocc-core**: Bindings Python per OpenCascade (CAD kernel)
- **numpy**: Calcoli numerici
- **Pillow**: Gestione immagini

## Utilizzo

### Avvio Applicazione

```bash
python main.py
```

### Workflow Base

1. **Apri un PDF**
   - File → Open PDF... (o Ctrl+O)
   - Seleziona un file PDF
   - Le linee verranno estratte automaticamente

2. **Naviga la Vista**
   - Zoom: Rotella del mouse
   - Pan: Pulsante centrale + drag
   - Zoom Extents: Premi F

3. **Modifica Geometria**
   - Seleziona oggetti cliccandoli
   - Usa toolbar per muovere, copiare o eliminare
   - Disegna nuove linee e punti

4. **Gestisci Layer**
   - Pannello Layer a destra
   - Crea nuovi layer
   - Assegna oggetti ai layer

## Struttura Progetto

```
Pycode/
├── main.py                 # Entry point applicazione
├── requirements.txt        # Dipendenze Python
├── README.md              # Documentazione
├── pdf_vectorizer/        # Modulo conversione PDF
│   ├── __init__.py
│   └── pdf_reader.py      # Estrazione vettoriale da PDF
├── cad_engine/            # Motore CAD
│   ├── __init__.py
│   ├── geometry.py        # Classi geometria (Point, Line)
│   ├── layer_manager.py   # Gestione layer
│   └── document.py        # Documento CAD principale
├── gui/                   # Interfaccia grafica
│   ├── __init__.py
│   ├── main_window.py     # Finestra principale
│   └── cad_viewport.py    # Viewport CAD con rendering
└── tools/                 # Strumenti CAD
    ├── __init__.py
    ├── selection.py       # Tool selezione
    ├── transform.py       # Tool trasformazione
    └── drawing.py         # Tool disegno
```

## Architettura

### Moduli Principali

1. **pdf_vectorizer**: Gestisce l'import e la conversione di file PDF in dati vettoriali
2. **cad_engine**: Core del sistema CAD con geometrie, layer e documento
3. **gui**: Interfaccia utente Qt con viewport e controlli
4. **tools**: Strumenti di editing (selezione, trasformazione, disegno)

### Flusso Dati

```
PDF File → PDFVectorizer → CAD Data → CADDocument → Viewport → Rendering
                              ↓
                         Layer Manager
                              ↓
                         Geometry Objects (Point, Line)
```

## Sviluppi Futuri

- Import/Export DXF, DWG
- Più primitive geometriche (cerchi, archi, spline)
- Snap e guide
- Misurazioni e quote
- Stampa e export immagini
- Supporto 3D completo con OpenCascade

## Licenza

Progetto educativo/sperimentale

## Autore

Sviluppato con Python, Qt e OpenCascade
