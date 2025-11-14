@echo off
echo ========================================
echo    AI Multi-Pair Monitor - Startup
echo ========================================
echo.
echo Avvio sistema AI di trading...
echo.

REM Controlla se Python e' installato
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato!
    echo Installa Python 3.7+ da https://www.python.org/
    pause
    exit /b 1
)

REM Controlla dipendenze
echo Verifica dipendenze...
pip show scikit-learn >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ATTENZIONE] scikit-learn non installato!
    echo Installazione dipendenze in corso...
    pip install -r requirements.txt
    echo.
)

REM Avvia il programma
echo.
echo Avvio AI Multi-Pair Monitor...
echo.
python ai_multi_pair_monitor.py

REM Se il programma termina con errore
if errorlevel 1 (
    echo.
    echo [ERRORE] Il programma e' terminato con errori.
    echo Controlla i messaggi sopra per dettagli.
    echo.
)

pause
