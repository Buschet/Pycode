@echo off
REM Script di installazione per Windows

echo =========================================
echo PDF to CAD Vectorizer - Installazione
echo =========================================
echo.

REM Verifica Python
echo Verificando Python...
python --version
if errorlevel 1 (
    echo ERRORE: Python non trovato!
    pause
    exit /b 1
)
echo OK Python trovato
echo.

REM Verifica pip
echo Verificando pip...
pip --version
if errorlevel 1 (
    echo ERRORE: pip non trovato!
    pause
    exit /b 1
)
echo OK pip trovato
echo.

REM Installazione dipendenze
echo Installando dipendenze Python...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo =========================================
    echo X Errore durante l'installazione
    echo =========================================
    pause
    exit /b 1
)

echo.
echo =========================================
echo OK Installazione completata con successo!
echo =========================================
echo.
echo Per avviare l'applicazione esegui:
echo   python main.py
echo.
pause
