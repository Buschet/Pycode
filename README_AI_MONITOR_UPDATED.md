# 🤖 AI Multi-Pair Monitor - VERSIONE AGGIORNATA

**Sistema Avanzato di Trading con Intelligenza Artificiale + TRADING AUTOMATICO**

Monitoraggio automatico di coppie **BTC/USDC** su Binance con ML e possibilità di esecuzione automatica dei trade.

---

## 🆕 NUOVE FEATURES (Versione 2.0)

### ✅ Filtro Coppie Ottimizzato
- **SOLO BTC e USDC** (rimosso USDT)
- Monitoraggio mirato sulle coppie più stabili
- Maggiore focus e performance

### ✅ Timeframe Configurabile
- **Selector grafico** con 7 timeframes:
  - `1m` - 1 minuto (scalping ultra-veloce)
  - `5m` - 5 minuti (scalping veloce)
  - `15m` - 15 minuti (scalping moderato) **[DEFAULT]**
  - `30m` - 30 minuti (intraday)
  - `1h` - 1 ora (swing trading)
  - `4h` - 4 ore (position trading)
  - `1d` - 1 giorno (trend following)

- **Candlestick data integration**: Il sistema ora scarica e analizza dati candlestick per il timeframe selezionato
- **Analisi multi-timeframe**: Features estratte considerano anche i pattern temporali

### ✅ Trading Automatico Integrato
- **Auto-trading con Binance** completamente integrato
- **Esecuzione automatica** dei segnali ML
- **Configurazione sicura** con dialog API
- **Trade amount configurabile** (10-500 USD)
- **Safety checks** multipli prima dell'esecuzione

---

## 🎮 Nuovi Controlli GUI

### Panel "Auto Trading" (4° colonna)

```
┌─────────────────────────────┐
│    🤖 Auto Trading          │
├─────────────────────────────┤
│ Status: Disabled            │  ← Status trading
│                             │
│ ☐ Enable Auto Trading      │  ← Checkbox attivazione
│                             │
│ Trade Amount ($): [50]      │  ← Amount per trade
│                             │
│ [⚙️ Setup API]              │  ← Config API Binance
└─────────────────────────────┘
```

### Timeframe Selector (3° colonna Config)

```
┌─────────────────────────────┐
│    ⚙️ Configuration         │
├─────────────────────────────┤
│ Update Interval (s): [10]   │
│ Pairs to Monitor: [50]      │
│ Min Confidence: [0.70]      │
│ Timeframe: [15m ▼]          │  ← NUOVO!
└─────────────────────────────┘
```

---

## 🚀 Setup Completo Passo-Passo

### 1️⃣ Installazione

```bash
pip install -r requirements.txt
```

### 2️⃣ Primo Avvio

```bash
python ai_multi_pair_monitor.py
```

### 3️⃣ Train del Modello ML

1. Clicca **"🎓 Train Model"** nel panel "ML Model"
2. Attendi training (~5-10 secondi)
3. Verifica accuracy (tipicamente 75-85%)
4. Clicca **"💾 Save Model"** per salvare

### 4️⃣ Setup API Binance (OPZIONALE - solo per auto-trading)

**Se NON vuoi auto-trading, salta questo step!**

1. Vai su [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Crea nuova API Key:
   - Nome: "AI Multi-Pair Monitor"
   - Permessi: **Enable Spot & Margin Trading**
   - IP Whitelist: Aggiungi il tuo IP pubblico (consigliato)
3. Copia API Key e Secret
4. Nel programma, clicca **"⚙️ Setup API"**
5. Incolla API Key e Secret
6. Clicca **"💾 Save & Connect"**
7. Verifica status: "Ready (Disabled)" in arancione

**⚠️ IMPORTANTE:**
- Le credenziali sono salvate in `binance_config.json`
- Aggiungi `binance_config.json` al tuo `.gitignore`
- NON condividere mai le tue API keys!

### 5️⃣ Carica le Coppie Trading

1. Clicca **"🔄 Refresh Pairs"**
2. Il sistema scaricherà tutte le coppie BTC/USDC disponibili
3. Vedrai il conteggio in basso (es. "Pairs: 125")

### 6️⃣ Configura Parametri

**Configuration Panel:**
- **Update Interval**: 10-20 secondi (consigliato)
- **Pairs to Monitor**: 30-50 coppie (per iniziare)
- **Min Confidence**: 0.70-0.75 (70-75%)
- **Timeframe**: 15m o 1h (consigliato per iniziare)

**Auto Trading Panel (se configurato API):**
- **Trade Amount**: 50-100 USD (conservativo)
- Lascia **checkbox disabilitata** per ora

### 7️⃣ Avvia Monitoring

1. Clicca **"🚀 Start Monitoring"**
2. Il sistema inizia a scansionare le coppie
3. Vedrai i segnali apparire nei tab "Signals Dashboard"

### 8️⃣ Abilita Auto-Trading (OPZIONALE)

**SOLO SE:**
- Hai configurato API Binance
- Hai testato il sistema in modalità monitor
- Sei sicuro di voler fare trading automatico

**Procedura:**
1. Verifica che il monitoring sia attivo
2. Spunta **"Enable Auto Trading"**
3. Leggi l'avviso di sicurezza
4. Clicca **"Yes"** se sei sicuro
5. Verifica status: "ACTIVE 🔥" in verde
6. Monitora i logs per vedere i trade eseguiti

**⚠️ STOP EMERGENZA:**
- Togli la spunta da "Enable Auto Trading"
- O clicca "⏹️ Stop" nel panel Monitoring

---

## 📊 Come Interpretare i Segnali

### Tab "Signals Dashboard"

**BUY Signals (Top 10):**
```
Symbol      Confidence  Price       Vol.Imb.  Momentum  Last Update
ETHUSDC     87.5%      $3,245.12   2.45x     +1.23%    14:32:15
ADAUSDC     82.1%      $0.4512     3.12x     +0.87%    14:32:14
```

**Interpretazione:**
- **ETHUSDC**: 87.5% confidence BUY
  - Volume imbalance: 2.45x (più bid che ask = pressione acquisto)
  - Momentum: +1.23% (trend positivo)
  - **Azione suggerita**: Considerare acquisto

**SELL Signals (Top 10):**
```
Symbol      Confidence  Price       Vol.Imb.  Momentum  Last Update
BTCUSDC     85.3%      $67,234.56  0.42x     -0.95%    14:32:13
```

**Interpretazione:**
- **BTCUSDC**: 85.3% confidence SELL
  - Volume imbalance: 0.42x (più ask che bid = pressione vendita)
  - Momentum: -0.95% (trend negativo)
  - **Azione suggerita**: Considerare vendita

---

## ⚙️ Configurazione Avanzata Timeframe

### Timeframe per Strategia

**Scalping (1m, 5m):**
- Pro: Segnali frequenti, opportunità multiple
- Contro: Molto noise, richiede attenzione costante
- Consigliato: Solo per esperti con bassa latency

**Intraday (15m, 30m):**
- Pro: Bilanciato signal/noise, gestibile
- Contro: Richiede monitoring regolare
- Consigliato: **Default per la maggior parte degli utenti**

**Swing Trading (1h, 4h):**
- Pro: Segnali più affidabili, meno noise
- Contro: Meno opportunità
- Consigliato: Per trading part-time

**Position Trading (1d):**
- Pro: Segnali molto affidabili, poco noise
- Contro: Pochissime opportunità
- Consigliato: Per investimenti a medio/lungo termine

### Come Scegliere il Timeframe

1. **Considera il tuo tempo disponibile:**
   - Poco tempo: 1h, 4h, 1d
   - Tempo moderato: 15m, 30m
   - Full-time: 5m, 15m

2. **Considera la tua esperienza:**
   - Principiante: 1h, 4h
   - Intermedio: 15m, 30m, 1h
   - Esperto: Qualsiasi

3. **Considera il capitale:**
   - Piccolo ($100-500): 15m, 1h
   - Medio ($500-2000): Qualsiasi
   - Grande ($2000+): 1h, 4h, 1d

---

## 🤖 Auto-Trading: Come Funziona

### Flusso di Esecuzione

```
1. Monitoring Loop
   ↓
2. Fetch orderbook + candlestick data (timeframe selezionato)
   ↓
3. Extract 20+ features
   ↓
4. ML Model predice: BUY / SELL / HOLD
   ↓
5. Calcola confidence score (0-100%)
   ↓
6. SE auto-trading ATTIVO E confidence >= min_confidence:
   ↓
7. Crea TradingSignal
   ↓
8. Binance Trading Integration verifica:
   - Balance disponibile
   - Max posizioni aperte
   - Simbolo consentito
   ↓
9. Esegue MARKET order su Binance
   ↓
10. Log risultato in GUI
```

### Safety Checks

**Prima dell'esecuzione:**
- ✅ Auto-trading abilitato
- ✅ Modello ML trained
- ✅ Confidence >= soglia minima
- ✅ Segnale != HOLD
- ✅ API Binance configurata
- ✅ Balance sufficiente
- ✅ Max posizioni non raggiunto
- ✅ Simbolo in allowed_quote_assets

**Se anche uno solo fallisce → Trade SKIPPED**

### Monitoraggio Trade Automatici

**Nel tab "Logs":**
```
[14:32:15] 🔍 Scanning 50 pairs...
[14:32:18] ✅ Scan completato (TF: 15m). Prossimo update in 10s | 🔥 AUTO TRADING ACTIVE
[14:32:18] ✅ AUTO TRADE: ETHUSDC BUY @ 87.5% confidence
[14:32:20] ⏸️ Trade skipped: ADAUSDC - Confidence too low: 0.68
[14:32:22] ❌ Trade failed: XRPUSDC - Insufficient balance
```

---

## 📈 Configurazione Consigliata per Diversi Profili

### 🐢 Principiante Conservativo

```yaml
Update Interval: 20s
Pairs to Monitor: 20-30
Min Confidence: 0.80 (80%)
Timeframe: 1h o 4h
Trade Amount: $50
Auto Trading: NO (solo monitor)
```

**Strategia:**
1. Monitora solo i segnali
2. Verifica su Binance manualmente
3. Esegui trade manuali quando sei sicuro
4. Impara a riconoscere i pattern

### 📊 Intermedio Bilanciato

```yaml
Update Interval: 15s
Pairs to Monitor: 50
Min Confidence: 0.75 (75%)
Timeframe: 15m o 30m
Trade Amount: $100
Auto Trading: SÌ (con attenzione)
```

**Strategia:**
1. Abilita auto-trading con importi bassi
2. Monitora i risultati costantemente
3. Tweaka min_confidence basandoti sui risultati
4. Aumenta gradualmente trade amount

### 🚀 Esperto Aggressivo

```yaml
Update Interval: 10s
Pairs to Monitor: 100+
Min Confidence: 0.70 (70%)
Timeframe: 5m o 15m
Trade Amount: $200-500
Auto Trading: SÌ
```

**Strategia:**
1. Auto-trading completamente attivo
2. Diversificazione su molte coppie
3. Lower confidence per più opportunità
4. Monitoring attivo dei risultati

---

## ⚠️ Disclaimer e Rischi

### IMPORTANTE - LEGGI PRIMA DI USARE AUTO-TRADING

1. **Rischio Perdita Capitale:**
   - Il trading comporta rischio di perdita
   - NON investire più di quanto puoi permetterti di perdere
   - I risultati passati non garantiscono performance future

2. **Sistema in BETA:**
   - Il software è in fase di test
   - Possibili bug e comportamenti imprevisti
   - Usa a tuo rischio

3. **Responsabilità:**
   - Tu sei l'UNICO responsabile dei trade eseguiti
   - L'autore non è responsabile per perdite
   - Verifica sempre i trade manualmente quando possibile

4. **Sicurezza API:**
   - Proteggi le tue API keys
   - Usa IP whitelist su Binance
   - NON condividere `binance_config.json`
   - Revoca keys se compromesse

5. **Rate Limiting:**
   - Rispetta i limiti API Binance
   - Evita configurazioni troppo aggressive
   - Rischio ban temporaneo se superi i limiti

6. **Volatilità Mercato:**
   - In mercati volatili, accuracy del modello può calare
   - Flash crashes possono causare perdite rapide
   - Monitora sempre durante eventi macro importanti

---

## 🔧 Troubleshooting

### "Binance Trading non configurato!"
**Soluzione:**
1. Clicca "Setup API"
2. Inserisci API Key e Secret
3. Clicca "Save & Connect"
4. Verifica che `advanced_binance_integration.py` esista

### "Trade failed: Insufficient balance"
**Soluzione:**
1. Verifica balance su Binance
2. Riduci "Trade Amount"
3. Controlla che asset USDC/BTC abbiano fondi

### "Trade skipped: Confidence too low"
**Soluzione:**
- Questo è normale!
- Significa che il segnale non raggiunge la confidence minima
- Riduci "Min Confidence" se vuoi più trade (non consigliato)

### Auto-trading non esegue nulla
**Verifica:**
1. Checkbox "Enable Auto Trading" è spuntata?
2. Status trading è "ACTIVE 🔥"?
3. Monitoring è avviato?
4. Ci sono segnali con confidence >= min?
5. Balance sufficiente?

### Troppi trade eseguiti
**Soluzione:**
1. Aumenta "Min Confidence" (es. 0.85)
2. Aumenta "Update Interval" (es. 30s)
3. Riduci "Pairs to Monitor"
4. Cambia timeframe a 1h o 4h

---

## 📊 Metriche e Performance

### Come Valutare il Modello

**Accuracy:**
- **>85%**: Eccellente
- **75-85%**: Buono (tipico)
- **<75%**: Da rivedere (ri-train)

**Win Rate Reale (dopo 50+ trade):**
- **>60%**: Ottimo
- **50-60%**: Buono
- **<50%**: Problematico (rivedi strategy)

### Logging e Analytics

**Tab Analytics:**
- Monitora distribuzione segnali
- Verifica confidence distribution
- Controlla volume imbalance patterns
- Traccia model performance

**Tab Logs:**
- Salva i logs in file per analisi
- Traccia tutti i trade eseguiti
- Debug problemi in tempo reale

---

## 🎓 Best Practices

### ✅ DO's

1. **Inizia in modalità monitor** (senza auto-trading)
2. **Testa con importi bassi** ($50-100)
3. **Monitora costantemente** i primi giorni
4. **Tweaka parametri** basandoti sui risultati
5. **Usa stop-loss** appropriati (già integrati)
6. **Diversifica** su più coppie
7. **Salva il modello** dopo ogni training
8. **Backup `binance_config.json`** in luogo sicuro
9. **Documenta** le tue configurazioni vincenti
10. **Review** periodico delle performance

### ❌ DON'Ts

1. **NON** usare tutto il capitale su un solo trade
2. **NON** ignorare i segnali di stop-loss
3. **NON** fare trading durante news macro importanti
4. **NON** aumentare trade amount dopo perdite (revenge trading)
5. **NON** usare timeframe troppo bassi (1m) senza esperienza
6. **NON** lasciare auto-trading attivo senza monitoring
7. **NON** condividere le tue API keys
8. **NON** usare confidence < 70% per auto-trading
9. **NON** fare over-trading (troppi trade ravvicinati)
10. **NON** dimenticare di fare backup del modello trained

---

## 📁 File di Configurazione

### binance_config.json
```json
{
  "api_key": "your_api_key_here",
  "api_secret": "your_api_secret_here"
}
```

**⚠️ Aggiungi al .gitignore:**
```bash
echo "binance_config.json" >> .gitignore
```

### ai_trading_model.pkl
- Modello ML salvato
- Riutilizzabile tra sessioni
- Backup consigliato

---

## 🆘 Support

### Per problemi tecnici:
1. Controlla questa documentazione
2. Verifica i logs nel tab "Logs"
3. Controlla requirements installati
4. Verifica connessione internet

### Per problemi Binance API:
1. Verifica API keys su Binance.com
2. Controlla IP whitelist
3. Verifica permessi API (Spot Trading)
4. Testa connettività API

---

## 📝 Changelog

### v2.0 (Corrente)
- ✅ Filtro solo BTC/USDC
- ✅ Timeframe selector (7 opzioni)
- ✅ Candlestick data integration
- ✅ Auto-trading Binance
- ✅ Setup API dialog
- ✅ Safety checks multipli
- ✅ Enhanced logging

### v1.0 (Precedente)
- Monitoring multi-coppia BTC/USDC/USDT
- ML model con 20+ features
- Dashboard analytics
- Solo analisi (no trading)

---

**Buon Trading Responsabile! 🚀📈**

*Remember: The best trader is the one who survives.*
