@echo off
REM ========================================
REM  🚀 TRADING SYSTEM LAUNCHER
REM  Sistema Avanzato di Trading Binance
REM ========================================

title Trading System - Binance Live Trading

echo.
echo ==========================================
echo  🚀 SISTEMA TRADING AVANZATO - LAUNCHER
echo ==========================================
echo.
echo 📊 Inizializzazione sistema...
echo 💰 Connessione a Binance...
echo 🎯 Caricamento strategie...
echo.

REM Cambia directory al percorso del tuo script
cd /d "C:\Users\User\Desktop\Ambiente"

REM Attiva l'ambiente virtuale Python (se utilizzato)
call env\Scripts\activate.bat

REM Avvia il sistema di trading
echo ✅ Avvio Trading System...
call python orderbook_limitorder_v0.py

REM Tieni aperta la finestra in caso di errore
echo.
echo ==========================================
echo  Sistema terminato. Premi un tasto per uscire...
echo ==========================================
pause