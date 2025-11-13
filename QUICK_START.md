# Quick Start Guide

## Installazione Rapida

### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### Windows
```cmd
install.bat
```

### Manuale
```bash
pip install -r requirements.txt
```

## Primo Avvio

```bash
python main.py
```

## Primi Passi

### 1. Aprire un PDF
- Click su "File" → "Open PDF..." (o Ctrl+O)
- Seleziona un file PDF con contenuto vettoriale
- Le linee verranno importate automaticamente

### 2. Navigare la Vista

| Azione | Metodo |
|--------|--------|
| **Zoom** | Rotella del mouse |
| **Pan** | Pulsante centrale mouse + drag |
| **Zoom Extents** | Tasto F o pulsante toolbar |
| **Ruota Vista** | Pulsanti toolbar ↶ ↷ |

### 3. Selezionare Oggetti
- Click sul pulsante "Select" nella toolbar
- Click su una linea o punto per selezionarlo
- Gli oggetti selezionati diventano gialli
- Ctrl+A per selezionare tutto

### 4. Modificare Geometria

| Tool | Azione |
|------|--------|
| **Move** | Seleziona + click tool Move + click destination |
| **Copy** | Seleziona + Ctrl+C (copia con offset 10,10) |
| **Delete** | Seleziona + Delete o pulsante Delete |

### 5. Disegnare Nuovi Elementi
- **Punto**: Click pulsante "Draw Point" + click sulla vista
- **Linea**: Click pulsante "Draw Line" + 2 click per start/end

### 6. Gestire Layer
- Pannello "Layers" a destra
- Click "New" per creare un nuovo layer
- Click su un layer per renderlo corrente
- Nuovi oggetti vengono creati sul layer corrente

## Scorciatoie Tastiera Utili

| Tasto | Funzione |
|-------|----------|
| F | Zoom extents |
| Esc | Annulla operazione corrente |
| Delete | Elimina selezione |
| Ctrl+A | Seleziona tutto |
| Ctrl+O | Apri PDF |
| Ctrl+C | Copia |
| Ctrl+Q | Esci |

## Risoluzione Problemi

### L'applicazione non si avvia
1. Verifica che le dipendenze siano installate:
   ```bash
   pip install -r requirements.txt
   ```

2. Verifica la versione di Python:
   ```bash
   python --version  # Deve essere >= 3.8
   ```

### Il PDF non viene caricato
- Assicurati che il PDF contenga contenuto vettoriale (non solo immagini)
- Prova con un PDF creato da CAD o programmi di disegno vettoriale

### Performance lente con molti oggetti
- Usa lo zoom per ridurre l'area visualizzata
- Disattiva layer non necessari
- Considera di semplificare il PDF prima dell'import

## Supporto

Per problemi o domande, controlla:
- README.md per documentazione completa
- test_imports.py per verificare l'installazione

## Tips & Tricks

1. **Zoom preciso**: Usa i pulsanti +/- nella toolbar per zoom controllato
2. **Selezione multipla**: Tieni premuto Ctrl mentre clicki su più oggetti (da implementare)
3. **Reset vista**: Se perdi la vista, premi F per zoom extents
4. **Layer organizzazione**: Usa layer diversi per tipo di elemento (es. "Contorni", "Quote", ecc.)

Buon lavoro! 🎨
