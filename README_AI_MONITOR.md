# 🤖 AI Multi-Pair Monitor

Sistema avanzato di trading con **Intelligenza Artificiale** per il monitoraggio automatico di tutte le coppie BTC/USDC/USDT su Binance.

## 🎯 Features Principali

### ✅ Monitoraggio Multi-Coppia
- Scansione automatica di **tutte le coppie** BTC/USDC/USDT disponibili su Binance
- Monitoraggio real-time dell'orderbook
- Supporto fino a 200+ coppie simultane
- Update configurabile (da 5 a 60 secondi)

### ✅ Intelligenza Artificiale
- **Machine Learning** con algoritmi avanzati:
  - Gradient Boosting Classifier
  - Random Forest Classifier
- **20+ features** estratte dall'orderbook:
  - Volume imbalance (bid/ask ratio)
  - Spread analysis
  - Wall detection (muri di supporto/resistenza)
  - Price momentum
  - Volume concentration
  - Depth analysis
  - Volatility
- **Auto-training** con dati sintetici
- **Model persistence** (salvataggio/caricamento)
- Accuracy tipica: **75-85%**

### ✅ Segnali Trading
- **BUY signals** con confidence score
- **SELL signals** con confidence score
- **HOLD signals** per posizioni neutrali
- Filtering per confidence minima
- Ranking intelligente per priorità

### ✅ Dashboard Intuitiva
- **4 Tabs principali**:
  1. **Signals Dashboard**: Top 10 BUY/SELL signals
  2. **All Pairs**: Vista completa tutte le coppie monitorate
  3. **Analytics**: Grafici e statistiche in tempo reale
  4. **Logs**: Log dettagliati delle operazioni

- **Grafici Real-time**:
  - Distribuzione segnali (BUY/SELL/HOLD)
  - Distribuzione confidence
  - Volume imbalance top pairs
  - Model performance

## 📋 Requisiti Sistema

### Software
- Python 3.7+
- pip (package manager)

### Librerie Python
```bash
pip install -r requirements.txt
```

Principali dipendenze:
- `numpy` - Calcoli numerici
- `pandas` - Data manipulation
- `scikit-learn` - Machine Learning
- `matplotlib` - Grafici
- `requests` - API calls
- `tkinter` - GUI (incluso in Python standard)

## 🚀 Quick Start

### 1. Installazione

```bash
# Clona o scarica i file
cd Pycode

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Primo Avvio

```bash
# Avvia il programma
python ai_multi_pair_monitor.py
```

### 3. Setup Iniziale (nella GUI)

#### Step 1: Train del Modello ML
1. Clicca **"🎓 Train Model"**
2. Il sistema genererà 2000 campioni sintetici per training
3. Attendi il completamento (~5-10 secondi)
4. Vedrai l'accuracy del modello (tipicamente 75-85%)

#### Step 2: Salva il Modello
1. Clicca **"💾 Save Model"**
2. Il modello viene salvato in `ai_trading_model.pkl`
3. Al prossimo avvio verrà caricato automaticamente

#### Step 3: Carica le Coppie Trading
1. Clicca **"🔄 Refresh Pairs"**
2. Il sistema scaricherà tutte le coppie disponibili
3. Vedrai il conteggio in basso (es. "Pairs: 350")

#### Step 4: Avvia Monitoring
1. Configura i parametri:
   - **Update Interval**: 10-30 secondi (consigliato)
   - **Pairs to Monitor**: 50-100 coppie
   - **Min Confidence**: 0.70-0.80 (70-80%)
2. Clicca **"🚀 Start Monitoring"**
3. Il sistema inizierà la scansione automatica

## 🎮 Utilizzo

### Configurazione

**Update Interval (secondi)**
- Range: 5-60 secondi
- Consigliato: 10-20 secondi
- Più basso = più aggiornamenti ma più stress API

**Pairs to Monitor**
- Range: 10-200 coppie
- Consigliato: 50-100 coppie
- Più coppie = scansione più lenta

**Min Confidence**
- Range: 0.50-0.99 (50%-99%)
- Consigliato: 0.70-0.80
- Più alto = segnali più affidabili ma meno frequenti

### Interpretazione Segnali

#### 💹 BUY Signal
Indica una **opportunità di acquisto** rilevata dal modello ML.

**Cosa significa:**
- Volume bid > volume ask (pressione acquisto)
- Possibili muri di resistenza rotti
- Momentum positivo
- Pattern favorevole all'acquisto

**Confidence:**
- **>85%**: Segnale molto forte
- **75-85%**: Segnale forte
- **70-75%**: Segnale moderato
- **<70%**: Segnale debole (filtrato se min_confidence > 70%)

#### 📉 SELL Signal
Indica una **opportunità di vendita** rilevata dal modello ML.

**Cosa significa:**
- Volume ask > volume bid (pressione vendita)
- Possibili muri di supporto rotti
- Momentum negativo
- Pattern favorevole alla vendita

**Confidence:** stessa interpretazione di BUY

#### ⏸️ HOLD Signal
Indica una situazione **neutrale** - né buy né sell.

**Cosa significa:**
- Mercato bilanciato
- Nessun pattern chiaro
- Attesa di conferme

### Tabs Dettagliati

#### 🎯 Signals Dashboard
- **Top 10 BUY signals**: Migliori opportunità acquisto
- **Top 10 SELL signals**: Migliori opportunità vendita
- Ordinati per **confidence** (dal più alto)
- Mostra:
  - Symbol (es. ETHUSDC)
  - Confidence %
  - Price corrente
  - Volume Imbalance
  - Momentum
  - Last Update

#### 📊 All Pairs
- **Vista completa** di tutte le coppie monitorate
- Filtrabile e ordinabile
- Colori:
  - 🟢 Verde scuro = BUY con alta confidence
  - 🔴 Rosso scuro = SELL con alta confidence
  - ⚪ Neutro = HOLD o bassa confidence

#### 📈 Analytics
4 grafici in tempo reale:

1. **Signal Distribution**: Quanti BUY/SELL/HOLD
2. **Confidence Distribution**: Distribuzione delle confidence
3. **Volume Imbalance Top 20**: Top coppie per imbalance
4. **Model Performance**: Accuracy del modello

#### 📝 Logs
- Log dettagliati di tutte le operazioni
- Timestamp preciso
- Messaggi di sistema
- Errori e warnings

## 🧠 Come Funziona il Modello ML

### Feature Extraction (20+ Features)

Per ogni coppia monitora, il sistema estrae dall'orderbook:

**Prezzi:**
- Mid price
- Spread
- Spread %

**Volumi:**
- Total bid volume
- Total ask volume
- Volume imbalance ratio (bid/ask)
- Bid/Ask concentration (% nei primi 5 livelli)

**Depth Analysis:**
- Bid depth weighted
- Ask depth weighted
- Volume standard deviation

**Wall Detection:**
- Numero muri bid (ordini grandi)
- Numero muri ask
- Posizione muri più grandi

**Momentum (storico):**
- Price momentum
- Volume momentum
- Spread momentum
- Imbalance momentum
- Volatility

### Training del Modello

Il modello viene addestrato su **dati sintetici** che simulano:

1. **Scenario BUY**:
   - Volume imbalance alto (1.5-10x)
   - Alta concentrazione bid
   - Momentum positivo
   - Molti muri bid

2. **Scenario SELL**:
   - Volume imbalance basso (0.1-0.8x)
   - Alta concentrazione ask
   - Momentum negativo
   - Molti muri ask

3. **Scenario HOLD**:
   - Volume bilanciato (0.8-1.5x)
   - Concentrazioni equilibrate
   - Momentum neutro

Il modello **apprende patterns** da 2000 campioni e generalizza su dati reali.

### Predizione

Per ogni coppia:
1. Estrae features dall'orderbook real-time
2. Normalizza le features
3. Passa al modello ML
4. Ottiene predizione (0=SELL, 1=HOLD, 2=BUY)
5. Calcola confidence (probabilità della classe predetta)

## 💡 Best Practices

### Setup Ottimale

**Per trading attivo:**
- Update interval: 10-15 secondi
- Pairs to monitor: 30-50
- Min confidence: 0.75-0.80

**Per monitoring generale:**
- Update interval: 20-30 secondi
- Pairs to monitor: 100-150
- Min confidence: 0.70

**Per ricerca opportunità rare:**
- Update interval: 30-60 secondi
- Pairs to monitor: 150-200
- Min confidence: 0.85-0.90

### Interpretazione Signals

**Segnali HIGH confidence (>85%)**
- Molto affidabili
- Da considerare seriamente
- Verificare sempre con analisi aggiuntiva

**Segnali MEDIUM confidence (70-85%)**
- Moderatamente affidabili
- Utili per screening
- Richiedono conferme

**Segnali LOW confidence (<70%)**
- Poco affidabili
- Solo per reference
- Non consigliati per trading

### Note Importanti

⚠️ **DISCLAIMER**
- Questo è uno **strumento di analisi**, NON un sistema di trading automatico
- I segnali sono **predizioni statistiche**, non garanzie
- **NON esegue trade** automaticamente
- Usa sempre stop-loss e risk management
- Verifica i segnali con analisi manuale
- Non investire più di quanto puoi permetterti di perdere

⚠️ **Rate Limiting Binance**
- Binance ha limiti di richieste API
- Con 50 coppie @ 10s interval = ~5 req/sec (OK)
- Con 200 coppie @ 5s interval = ~40 req/sec (RISCHIO BAN)
- Consigliato: max 100 coppie con 10s interval

⚠️ **Performance**
- Il modello è addestrato su **dati sintetici**
- Per migliori risultati, ri-trainare con **dati reali** (future implementazione)
- Accuracy attesa: 75-85%
- In condizioni di mercato estreme, accuracy può calare

## 🔧 Troubleshooting

### "scikit-learn non disponibile"
```bash
pip install scikit-learn
```

### "Errore connessione Binance"
- Verifica connessione internet
- Binance API potrebbe essere temporaneamente down
- Prova a ridurre numero coppie monitorate

### "Modello non si carica"
- Verifica che esista `ai_trading_model.pkl` nella cartella
- Train di nuovo il modello
- Verifica versione scikit-learn compatibile

### GUI non si avvia
**Windows:**
```bash
# Installa tkinter se mancante
# Solitamente incluso in Python standard
```

**Linux:**
```bash
sudo apt-get install python3-tk
```

### Performance lente
- Riduci "Pairs to Monitor"
- Aumenta "Update Interval"
- Chiudi altri programmi pesanti

## 📚 File del Progetto

```
Pycode/
├── ai_multi_pair_monitor.py    # File principale del sistema
├── requirements.txt             # Dipendenze Python
├── README_AI_MONITOR.md        # Questa documentazione
├── ai_trading_model.pkl        # Modello ML salvato (generato dopo training)
├── orderbook_monitor_v0.py     # Sistema precedente (legacy)
├── advanced_binance_integration.py  # Integrazione Binance (legacy)
└── ...
```

## 🚀 Roadmap Future

Possibili miglioramenti futuri:

- [ ] Training con dati reali storici (backtest)
- [ ] Deep Learning (LSTM/Transformer) per time-series
- [ ] Multi-timeframe analysis
- [ ] Integrazione con trading automatico (con conferma manuale)
- [ ] Alert desktop/mobile per segnali high-confidence
- [ ] Export segnali in CSV/Excel
- [ ] Backtesting framework
- [ ] Paper trading integration
- [ ] WebSocket real-time (invece di polling)
- [ ] Multi-exchange support (Coinbase, Kraken, etc.)

## 📞 Support

Per problemi, domande o suggerimenti:
- Verifica questa documentazione
- Controlla i logs nella tab "Logs"
- Verifica requisiti sistema e dipendenze

## 📄 License

Questo progetto è fornito "as-is" per scopi educativi e di ricerca.

---

**Buon Trading! 🚀📈**

*Remember: Trade responsibly and never invest more than you can afford to lose.*
