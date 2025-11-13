#!/bin/bash
# Script di installazione per PDF to CAD Vectorizer

echo "========================================="
echo "PDF to CAD Vectorizer - Installazione"
echo "========================================="
echo ""

# Verifica Python
echo "Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERRORE: Python 3 non trovato!"
    exit 1
fi
echo "✓ Python trovato"
echo ""

# Verifica pip
echo "Verificando pip..."
pip --version
if [ $? -ne 0 ]; then
    echo "ERRORE: pip non trovato!"
    exit 1
fi
echo "✓ pip trovato"
echo ""

# Installazione dipendenze
echo "Installando dipendenze Python..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Installazione completata con successo!"
    echo "========================================="
    echo ""
    echo "Per avviare l'applicazione esegui:"
    echo "  python main.py"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Errore durante l'installazione"
    echo "========================================="
    exit 1
fi
