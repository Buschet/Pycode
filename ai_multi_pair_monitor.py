#!/usr/bin/env python3
"""
AI Multi-Pair Monitor - Sistema Avanzato di Trading con Intelligenza Artificiale
Monitoraggio automatico di tutte le coppie BTC/USDC con ML per segnali buy/sell
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
from collections import deque
from typing import Dict, List, Tuple
import json
import pickle
import os

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn non disponibile. Installa con: pip install scikit-learn")

# Grafica
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.style.use('dark_background')


class OrderBookFeatureExtractor:
    """Estrae features avanzate dall'orderbook per il modello ML"""

    @staticmethod
    def extract_features(orderbook: dict, historical_data: deque = None) -> dict:
        """Estrae 20+ features dall'orderbook"""

        if not orderbook.get('bids') or not orderbook.get('asks'):
            return None

        bids = orderbook['bids'][:20]
        asks = orderbook['asks'][:20]

        # Converti in numpy arrays
        bid_prices = np.array([float(p) for p, v in bids])
        bid_volumes = np.array([float(v) for p, v in bids])
        ask_prices = np.array([float(p) for p, v in asks])
        ask_volumes = np.array([float(v) for p, v in asks])

        # === FEATURE 1-5: Prezzi base ===
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100

        # === FEATURE 6-10: Volumi ===
        total_bid_volume = bid_volumes.sum()
        total_ask_volume = ask_volumes.sum()
        total_volume = total_bid_volume + total_ask_volume

        # Volume imbalance (ratio bid/ask)
        volume_imbalance = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 0

        # Volume concentration (% nei primi 5 livelli)
        bid_concentration = bid_volumes[:5].sum() / total_bid_volume if total_bid_volume > 0 else 0
        ask_concentration = ask_volumes[:5].sum() / total_ask_volume if total_ask_volume > 0 else 0

        # === FEATURE 11-15: Depth Analysis ===
        # Profondità media ponderata
        bid_depth_weighted = np.average(bid_prices, weights=bid_volumes) if total_bid_volume > 0 else best_bid
        ask_depth_weighted = np.average(ask_prices, weights=ask_volumes) if total_ask_volume > 0 else best_ask

        # Distanza depth weighted dal mid price
        bid_depth_distance = (mid_price - bid_depth_weighted) / mid_price * 100
        ask_depth_distance = (ask_depth_weighted - mid_price) / mid_price * 100

        # Skewness del volume
        bid_volume_std = bid_volumes.std()
        ask_volume_std = ask_volumes.std()

        # === FEATURE 16-20: Wall Detection ===
        # Identifica "muri" (ordini > 2x volume medio)
        avg_bid_volume = bid_volumes.mean()
        avg_ask_volume = ask_volumes.mean()

        bid_walls = (bid_volumes > avg_bid_volume * 2).sum()
        ask_walls = (ask_volumes > avg_ask_volume * 2).sum()

        # Posizione dei muri
        largest_bid_idx = bid_volumes.argmax()
        largest_ask_idx = ask_volumes.argmax()

        # === FEATURE 21-25: Momentum (se disponibile storico) ===
        price_momentum = 0
        volume_momentum = 0
        spread_momentum = 0
        imbalance_momentum = 0
        volatility = 0

        if historical_data and len(historical_data) >= 5:
            recent = list(historical_data)[-5:]
            prices = [h['mid_price'] for h in recent]
            volumes = [h['total_volume'] for h in recent]
            spreads = [h['spread_pct'] for h in recent]
            imbalances = [h['volume_imbalance'] for h in recent]

            # Momentum come % change
            price_momentum = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
            volume_momentum = (volumes[-1] - volumes[0]) / volumes[0] * 100 if volumes[0] > 0 else 0
            spread_momentum = (spreads[-1] - spreads[0]) if len(spreads) > 0 else 0
            imbalance_momentum = (imbalances[-1] - imbalances[0]) if len(imbalances) > 0 else 0

            # Volatilità
            volatility = np.std(prices) / np.mean(prices) * 100 if np.mean(prices) > 0 else 0

        # === Return feature dict ===
        features = {
            # Prezzi
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,

            # Volumi
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'total_volume': total_volume,
            'volume_imbalance': volume_imbalance,
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration,

            # Depth
            'bid_depth_distance': bid_depth_distance,
            'ask_depth_distance': ask_depth_distance,
            'bid_volume_std': bid_volume_std,
            'ask_volume_std': ask_volume_std,

            # Walls
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'largest_bid_position': largest_bid_idx,
            'largest_ask_position': largest_ask_idx,

            # Momentum
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'spread_momentum': spread_momentum,
            'imbalance_momentum': imbalance_momentum,
            'volatility': volatility,

            # Metadata
            'timestamp': datetime.now()
        }

        return features


class MLTradingModel:
    """Modello di Machine Learning per previsioni buy/sell"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.accuracy = 0.0

    def prepare_training_data(self, features_list: List[dict], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dati per training"""

        # Features numeriche (escludi timestamp e metadati)
        feature_keys = [
            'spread_pct', 'total_bid_volume', 'total_ask_volume', 'total_volume',
            'volume_imbalance', 'bid_concentration', 'ask_concentration',
            'bid_depth_distance', 'ask_depth_distance', 'bid_volume_std', 'ask_volume_std',
            'bid_walls', 'ask_walls', 'largest_bid_position', 'largest_ask_position',
            'price_momentum', 'volume_momentum', 'spread_momentum', 'imbalance_momentum', 'volatility'
        ]

        self.feature_names = feature_keys

        # Converti in array
        X = []
        for features in features_list:
            row = [features.get(k, 0) for k in feature_keys]
            X.append(row)

        X = np.array(X)
        y = np.array(labels)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train del modello con validation"""

        print(f"\n🤖 TRAINING ML MODEL...")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalizza features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Valutazione
        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)

        print(f"   ✅ Accuracy: {self.accuracy*100:.2f}%")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n   🔝 Top 5 Features:")
        for feat, imp in top_features:
            print(f"      {feat}: {imp:.4f}")

        self.is_trained = True
        return self.accuracy

    def predict(self, features: dict) -> Tuple[int, float]:
        """Predizione: 0=SELL, 1=HOLD, 2=BUY + confidence"""

        if not self.is_trained or not self.model:
            return 1, 0.33  # HOLD con bassa confidence

        # Prepara features
        X = np.array([[features.get(k, 0) for k in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # Predizione
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        confidence = probabilities[prediction]

        return int(prediction), float(confidence)

    def save_model(self, filepath: str):
        """Salva modello su disco"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'accuracy': self.accuracy
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"💾 Modello salvato: {filepath}")

    def load_model(self, filepath: str):
        """Carica modello da disco"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            self.is_trained = True

            print(f"✅ Modello caricato: {filepath} (Accuracy: {self.accuracy*100:.2f}%)")
            return True
        return False


class BinanceMultiPairMonitor:
    """Monitor multi-coppia con AI"""

    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.pairs_data = {}  # {symbol: {orderbook, features, historical_data}}
        self.ml_model = MLTradingModel() if ML_AVAILABLE else None

        # Cache simboli
        self.btc_pairs = []
        self.usdc_pairs = []
        self.all_trading_pairs = []

        # Thread safety
        self.lock = threading.Lock()

    def fetch_all_trading_pairs(self, base_assets: List[str] = ['BTC', 'USDC', 'USDT']) -> List[str]:
        """Fetch tutte le coppie disponibili"""

        try:
            response = requests.get(f"{self.base_url}/api/v3/exchangeInfo", timeout=10)

            if response.status_code == 200:
                data = response.json()

                pairs = []
                for symbol_info in data['symbols']:
                    if symbol_info['status'] == 'TRADING':
                        symbol = symbol_info['symbol']
                        quote = symbol_info['quoteAsset']

                        # Filtra per base asset richiesti
                        if quote in base_assets:
                            pairs.append(symbol)

                            # Categorizza
                            if quote == 'BTC':
                                self.btc_pairs.append(symbol)
                            elif quote in ['USDC', 'USDT']:
                                self.usdc_pairs.append(symbol)

                self.all_trading_pairs = pairs

                print(f"✅ Trovate {len(pairs)} coppie trading")
                print(f"   BTC pairs: {len(self.btc_pairs)}")
                print(f"   USDC/USDT pairs: {len(self.usdc_pairs)}")

                return pairs

        except Exception as e:
            print(f"❌ Errore fetch pairs: {e}")
            return []

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Fetch orderbook per simbolo"""

        try:
            url = f"{self.base_url}/api/v3/depth"
            params = {'symbol': symbol, 'limit': limit}

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                return {
                    'bids': [[float(p), float(v)] for p, v in data.get('bids', [])],
                    'asks': [[float(p), float(v)] for p, v in data.get('asks', [])],
                    'timestamp': datetime.now()
                }

        except Exception as e:
            print(f"❌ Error fetch {symbol}: {e}")

        return {'bids': [], 'asks': [], 'timestamp': datetime.now()}

    def update_pair_data(self, symbol: str):
        """Aggiorna dati per una coppia"""

        with self.lock:
            # Inizializza se non esiste
            if symbol not in self.pairs_data:
                self.pairs_data[symbol] = {
                    'orderbook': None,
                    'features': None,
                    'historical_features': deque(maxlen=50),
                    'prediction': None,
                    'confidence': 0.0,
                    'last_update': None
                }

            pair_data = self.pairs_data[symbol]

            # Fetch orderbook
            orderbook = self.fetch_orderbook(symbol)
            pair_data['orderbook'] = orderbook

            # Estrai features
            extractor = OrderBookFeatureExtractor()
            features = extractor.extract_features(orderbook, pair_data['historical_features'])

            if features:
                pair_data['features'] = features
                pair_data['historical_features'].append(features)

                # Predizione ML
                if self.ml_model and self.ml_model.is_trained:
                    prediction, confidence = self.ml_model.predict(features)
                    pair_data['prediction'] = prediction  # 0=SELL, 1=HOLD, 2=BUY
                    pair_data['confidence'] = confidence

                pair_data['last_update'] = datetime.now()

    def get_top_signals(self, n: int = 10, signal_type: str = 'BUY') -> List[Tuple[str, dict]]:
        """Ottieni top N segnali per tipo"""

        signal_map = {'BUY': 2, 'HOLD': 1, 'SELL': 0}
        target_signal = signal_map.get(signal_type, 2)

        with self.lock:
            # Filtra coppie con predizione matching
            matching_pairs = [
                (symbol, data)
                for symbol, data in self.pairs_data.items()
                if data.get('prediction') == target_signal
            ]

            # Ordina per confidence
            sorted_pairs = sorted(
                matching_pairs,
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )

            return sorted_pairs[:n]

    def generate_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[List[dict], List[int]]:
        """Genera dati sintetici per training iniziale"""

        print(f"\n🔬 Generando {n_samples} campioni sintetici per training...")

        features_list = []
        labels = []

        for _ in range(n_samples):
            # Parametri randomici per diversi scenari di mercato

            # BUY scenario (label=2): volume imbalance alto, bid concentration alta
            # SELL scenario (label=0): volume imbalance basso, ask concentration alta
            # HOLD scenario (label=1): bilanciato

            scenario = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])  # SELL, HOLD, BUY

            if scenario == 2:  # BUY
                volume_imbalance = np.random.uniform(1.5, 10.0)
                bid_concentration = np.random.uniform(0.6, 0.9)
                ask_concentration = np.random.uniform(0.3, 0.6)
                price_momentum = np.random.uniform(-0.5, 2.0)
                bid_walls = np.random.randint(2, 8)
                ask_walls = np.random.randint(0, 3)

            elif scenario == 0:  # SELL
                volume_imbalance = np.random.uniform(0.1, 0.8)
                bid_concentration = np.random.uniform(0.3, 0.6)
                ask_concentration = np.random.uniform(0.6, 0.9)
                price_momentum = np.random.uniform(-2.0, 0.5)
                bid_walls = np.random.randint(0, 3)
                ask_walls = np.random.randint(2, 8)

            else:  # HOLD
                volume_imbalance = np.random.uniform(0.8, 1.5)
                bid_concentration = np.random.uniform(0.4, 0.7)
                ask_concentration = np.random.uniform(0.4, 0.7)
                price_momentum = np.random.uniform(-0.5, 0.5)
                bid_walls = np.random.randint(0, 4)
                ask_walls = np.random.randint(0, 4)

            # Features complete
            features = {
                'spread_pct': np.random.uniform(0.01, 0.5),
                'total_bid_volume': np.random.uniform(100, 10000),
                'total_ask_volume': np.random.uniform(100, 10000),
                'total_volume': np.random.uniform(200, 20000),
                'volume_imbalance': volume_imbalance,
                'bid_concentration': bid_concentration,
                'ask_concentration': ask_concentration,
                'bid_depth_distance': np.random.uniform(0.1, 2.0),
                'ask_depth_distance': np.random.uniform(0.1, 2.0),
                'bid_volume_std': np.random.uniform(10, 500),
                'ask_volume_std': np.random.uniform(10, 500),
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'largest_bid_position': np.random.randint(0, 10),
                'largest_ask_position': np.random.randint(0, 10),
                'price_momentum': price_momentum,
                'volume_momentum': np.random.uniform(-5, 5),
                'spread_momentum': np.random.uniform(-0.1, 0.1),
                'imbalance_momentum': np.random.uniform(-2, 2),
                'volatility': np.random.uniform(0.1, 5.0)
            }

            features_list.append(features)
            labels.append(scenario)

        print(f"   ✅ Generati {len(features_list)} campioni")
        print(f"   Distribuzione: BUY={labels.count(2)}, HOLD={labels.count(1)}, SELL={labels.count(0)}")

        return features_list, labels


class AIMultiPairGUI:
    """Interfaccia grafica principale"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 AI Multi-Pair Monitor - Binance Trading Intelligence")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1a1a1a')

        # Monitor e ML model
        self.monitor = BinanceMultiPairMonitor()
        self.is_monitoring = False
        self.monitoring_thread = None

        # Config
        self.config = {
            'update_interval': 10,  # secondi
            'pairs_to_monitor': 50,  # numero coppie da monitorare
            'min_confidence': 0.70,  # confidence minima per segnale
            'base_assets': ['USDC', 'USDT']  # Asset base da monitorare
        }

        # Setup GUI
        self.setup_gui()

        # Auto-load model se esiste
        model_path = 'ai_trading_model.pkl'
        if os.path.exists(model_path):
            self.monitor.ml_model.load_model(model_path)
            self.update_model_status()

    def setup_gui(self):
        """Setup interfaccia principale"""

        # === HEADER ===
        header_frame = tk.Frame(self.root, bg='#2d2d2d', pady=10)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        title_label = tk.Label(
            header_frame,
            text="🤖 AI Multi-Pair Monitor",
            font=('Arial', 20, 'bold'),
            bg='#2d2d2d',
            fg='#00ff00'
        )
        title_label.pack()

        subtitle = tk.Label(
            header_frame,
            text="Sistema di Trading Intelligence con Machine Learning",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#888888'
        )
        subtitle.pack()

        # === CONTROL PANEL ===
        control_frame = tk.Frame(self.root, bg='#2d2d2d', pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Colonna 1: ML Model
        ml_frame = tk.LabelFrame(control_frame, text="🧠 ML Model", bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold'))
        ml_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        self.model_status_label = tk.Label(ml_frame, text="Status: Not Trained", bg='#2d2d2d', fg='red', font=('Arial', 9))
        self.model_status_label.pack(pady=2)

        tk.Button(ml_frame, text="🎓 Train Model", command=self.train_model, bg='#4CAF50', fg='white', width=15).pack(pady=2)
        tk.Button(ml_frame, text="💾 Save Model", command=self.save_model, bg='#2196F3', fg='white', width=15).pack(pady=2)
        tk.Button(ml_frame, text="📂 Load Model", command=self.load_model, bg='#FF9800', fg='white', width=15).pack(pady=2)

        # Colonna 2: Monitoring
        monitor_frame = tk.LabelFrame(control_frame, text="📡 Monitoring", bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold'))
        monitor_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        self.monitor_status_label = tk.Label(monitor_frame, text="Status: Stopped", bg='#2d2d2d', fg='red', font=('Arial', 9))
        self.monitor_status_label.pack(pady=2)

        self.start_btn = tk.Button(monitor_frame, text="🚀 Start Monitoring", command=self.start_monitoring, bg='#4CAF50', fg='white', width=15)
        self.start_btn.pack(pady=2)

        self.stop_btn = tk.Button(monitor_frame, text="⏹️ Stop", command=self.stop_monitoring, bg='#f44336', fg='white', width=15, state='disabled')
        self.stop_btn.pack(pady=2)

        tk.Button(monitor_frame, text="🔄 Refresh Pairs", command=self.refresh_pairs, bg='#2196F3', fg='white', width=15).pack(pady=2)

        # Colonna 3: Config
        config_frame = tk.LabelFrame(control_frame, text="⚙️ Configuration", bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold'))
        config_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        tk.Label(config_frame, text="Update Interval (s):", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack()
        self.interval_var = tk.IntVar(value=self.config['update_interval'])
        tk.Spinbox(config_frame, from_=5, to=60, textvariable=self.interval_var, width=10).pack(pady=2)

        tk.Label(config_frame, text="Pairs to Monitor:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack()
        self.pairs_var = tk.IntVar(value=self.config['pairs_to_monitor'])
        tk.Spinbox(config_frame, from_=10, to=200, increment=10, textvariable=self.pairs_var, width=10).pack(pady=2)

        tk.Label(config_frame, text="Min Confidence:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack()
        self.confidence_var = tk.DoubleVar(value=self.config['min_confidence'])
        tk.Spinbox(config_frame, from_=0.5, to=0.99, increment=0.05, textvariable=self.confidence_var, width=10).pack(pady=2)

        # === NOTEBOOK TABS ===
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Signals Dashboard
        self.setup_signals_tab(notebook)

        # Tab 2: All Pairs
        self.setup_pairs_tab(notebook)

        # Tab 3: Analytics
        self.setup_analytics_tab(notebook)

        # Tab 4: Logs
        self.setup_logs_tab(notebook)

        # === STATUS BAR ===
        status_frame = tk.Frame(self.root, bg='#2d2d2d', pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(status_frame, text="Ready", bg='#2d2d2d', fg='#00ff00', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.pairs_count_label = tk.Label(status_frame, text="Pairs: 0", bg='#2d2d2d', fg='white', font=('Arial', 9))
        self.pairs_count_label.pack(side=tk.RIGHT, padx=10)

    def setup_signals_tab(self, notebook):
        """Tab segnali buy/sell"""
        frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(frame, text="🎯 Signals Dashboard")

        # Top frame: BUY signals
        buy_frame = tk.LabelFrame(frame, text="💹 BUY Signals (Top 10)", bg='#1a1a1a', fg='#00ff00', font=('Arial', 12, 'bold'))
        buy_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ('Symbol', 'Confidence', 'Price', 'Volume Imbalance', 'Momentum', 'Last Update')
        self.buy_tree = ttk.Treeview(buy_frame, columns=columns, show='headings', height=10)

        for col in columns:
            self.buy_tree.heading(col, text=col)
            width = 120 if col == 'Symbol' else 150
            self.buy_tree.column(col, width=width)

        buy_scroll = ttk.Scrollbar(buy_frame, orient=tk.VERTICAL, command=self.buy_tree.yview)
        self.buy_tree.configure(yscrollcommand=buy_scroll.set)

        self.buy_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        buy_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom frame: SELL signals
        sell_frame = tk.LabelFrame(frame, text="📉 SELL Signals (Top 10)", bg='#1a1a1a', fg='#ff4444', font=('Arial', 12, 'bold'))
        sell_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.sell_tree = ttk.Treeview(sell_frame, columns=columns, show='headings', height=10)

        for col in columns:
            self.sell_tree.heading(col, text=col)
            width = 120 if col == 'Symbol' else 150
            self.sell_tree.column(col, width=width)

        sell_scroll = ttk.Scrollbar(sell_frame, orient=tk.VERTICAL, command=self.sell_tree.yview)
        self.sell_tree.configure(yscrollcommand=sell_scroll.set)

        self.sell_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sell_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_pairs_tab(self, notebook):
        """Tab tutte le coppie"""
        frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(frame, text="📊 All Pairs")

        columns = ('Symbol', 'Signal', 'Confidence', 'Price', 'Spread %', 'Volume Imb.', 'Momentum', 'Status')
        self.pairs_tree = ttk.Treeview(frame, columns=columns, show='headings')

        for col in columns:
            self.pairs_tree.heading(col, text=col)
            width = 100 if col == 'Symbol' else 120
            self.pairs_tree.column(col, width=width)

        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.pairs_tree.yview)
        self.pairs_tree.configure(yscrollcommand=scroll.set)

        self.pairs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_analytics_tab(self, notebook):
        """Tab analytics e grafici"""
        frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(frame, text="📈 Analytics")

        # Matplotlib figure
        self.fig = plt.Figure(figsize=(12, 8), facecolor='#1a1a1a')

        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')

        self.ax1.set_title('Signal Distribution', color='white')
        self.ax2.set_title('Confidence Distribution', color='white')
        self.ax3.set_title('Volume Imbalance Top 20', color='white')
        self.ax4.set_title('Model Performance', color='white')

        self.fig.tight_layout()

        canvas = FigureCanvasTkAgg(self.fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def setup_logs_tab(self, notebook):
        """Tab logs"""
        frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(frame, text="📝 Logs")

        self.log_text = scrolledtext.ScrolledText(frame, bg='#2d2d2d', fg='#00ff00', font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log("🚀 AI Multi-Pair Monitor inizializzato")
        self.log(f"💡 Machine Learning: {'✅ Disponibile' if ML_AVAILABLE else '❌ Non disponibile'}")

    def log(self, message: str):
        """Log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def train_model(self):
        """Train ML model con dati sintetici"""

        if not ML_AVAILABLE:
            messagebox.showerror("Error", "scikit-learn non disponibile!\nInstalla con: pip install scikit-learn")
            return

        self.log("🎓 Avvio training modello ML...")

        # Genera dati sintetici
        features_list, labels = self.monitor.generate_synthetic_training_data(n_samples=2000)

        # Prepara dati
        X, y = self.monitor.ml_model.prepare_training_data(features_list, labels)

        # Train
        accuracy = self.monitor.ml_model.train(X, y)

        self.log(f"✅ Training completato! Accuracy: {accuracy*100:.2f}%")

        self.update_model_status()

        messagebox.showinfo("Training Complete", f"Modello addestrato con successo!\n\nAccuracy: {accuracy*100:.2f}%")

    def save_model(self):
        """Salva modello"""
        if self.monitor.ml_model and self.monitor.ml_model.is_trained:
            self.monitor.ml_model.save_model('ai_trading_model.pkl')
            self.log("💾 Modello salvato: ai_trading_model.pkl")
            messagebox.showinfo("Save", "Modello salvato con successo!")
        else:
            messagebox.showwarning("Warning", "Nessun modello da salvare. Train prima il modello!")

    def load_model(self):
        """Carica modello"""
        if self.monitor.ml_model and self.monitor.ml_model.load_model('ai_trading_model.pkl'):
            self.log("📂 Modello caricato: ai_trading_model.pkl")
            self.update_model_status()
            messagebox.showinfo("Load", "Modello caricato con successo!")
        else:
            messagebox.showerror("Error", "Impossibile caricare il modello!")

    def update_model_status(self):
        """Aggiorna status modello"""
        if self.monitor.ml_model and self.monitor.ml_model.is_trained:
            self.model_status_label.config(
                text=f"Status: Trained (Acc: {self.monitor.ml_model.accuracy*100:.2f}%)",
                fg='#00ff00'
            )
        else:
            self.model_status_label.config(text="Status: Not Trained", fg='red')

    def refresh_pairs(self):
        """Refresh lista coppie"""
        self.log("🔄 Fetching trading pairs...")

        base_assets = self.config['base_assets']
        pairs = self.monitor.fetch_all_trading_pairs(base_assets)

        self.log(f"✅ Trovate {len(pairs)} coppie trading")
        self.pairs_count_label.config(text=f"Pairs: {len(pairs)}")

        messagebox.showinfo("Pairs Loaded", f"Caricate {len(pairs)} coppie di trading!")

    def start_monitoring(self):
        """Avvia monitoring"""

        if not self.monitor.ml_model or not self.monitor.ml_model.is_trained:
            messagebox.showwarning("Warning", "Modello ML non addestrato!\nTrain prima il modello.")
            return

        if not self.monitor.all_trading_pairs:
            self.refresh_pairs()

        self.is_monitoring = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.monitor_status_label.config(text="Status: Running", fg='#00ff00')

        self.log("🚀 Monitoring avviato!")

        # Avvia thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Ferma monitoring"""
        self.is_monitoring = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.monitor_status_label.config(text="Status: Stopped", fg='red')

        self.log("⏹️ Monitoring fermato")

    def monitoring_loop(self):
        """Loop principale monitoring"""

        while self.is_monitoring:
            try:
                # Update config
                self.config['update_interval'] = self.interval_var.get()
                self.config['pairs_to_monitor'] = self.pairs_var.get()
                self.config['min_confidence'] = self.confidence_var.get()

                # Seleziona coppie da monitorare
                pairs_to_check = self.monitor.all_trading_pairs[:self.config['pairs_to_monitor']]

                self.log(f"🔍 Scanning {len(pairs_to_check)} pairs...")

                # Aggiorna dati per ogni coppia
                for symbol in pairs_to_check:
                    if not self.is_monitoring:
                        break

                    self.monitor.update_pair_data(symbol)

                # Aggiorna GUI
                self.root.after(0, self.update_signals_display)
                self.root.after(0, self.update_pairs_display)
                self.root.after(0, self.update_analytics)

                self.log(f"✅ Scan completato. Prossimo update in {self.config['update_interval']}s")

                # Sleep
                time.sleep(self.config['update_interval'])

            except Exception as e:
                self.log(f"❌ Errore monitoring loop: {e}")
                time.sleep(5)

    def update_signals_display(self):
        """Aggiorna display segnali"""

        # Clear
        for item in self.buy_tree.get_children():
            self.buy_tree.delete(item)
        for item in self.sell_tree.get_children():
            self.sell_tree.delete(item)

        min_conf = self.config['min_confidence']

        # BUY signals
        buy_signals = self.monitor.get_top_signals(n=10, signal_type='BUY')
        for symbol, data in buy_signals:
            if data.get('confidence', 0) >= min_conf:
                features = data.get('features', {})

                values = (
                    symbol,
                    f"{data.get('confidence', 0)*100:.1f}%",
                    f"${features.get('mid_price', 0):.6f}",
                    f"{features.get('volume_imbalance', 0):.2f}x",
                    f"{features.get('price_momentum', 0):+.2f}%",
                    data.get('last_update', datetime.now()).strftime('%H:%M:%S')
                )

                self.buy_tree.insert('', tk.END, values=values)

        # SELL signals
        sell_signals = self.monitor.get_top_signals(n=10, signal_type='SELL')
        for symbol, data in sell_signals:
            if data.get('confidence', 0) >= min_conf:
                features = data.get('features', {})

                values = (
                    symbol,
                    f"{data.get('confidence', 0)*100:.1f}%",
                    f"${features.get('mid_price', 0):.6f}",
                    f"{features.get('volume_imbalance', 0):.2f}x",
                    f"{features.get('price_momentum', 0):+.2f}%",
                    data.get('last_update', datetime.now()).strftime('%H:%M:%S')
                )

                self.sell_tree.insert('', tk.END, values=values)

    def update_pairs_display(self):
        """Aggiorna display tutte coppie"""

        # Clear
        for item in self.pairs_tree.get_children():
            self.pairs_tree.delete(item)

        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        # Sort by confidence
        pairs_sorted = sorted(
            self.monitor.pairs_data.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )

        for symbol, data in pairs_sorted[:100]:  # Top 100
            features = data.get('features', {})
            prediction = data.get('prediction')
            confidence = data.get('confidence', 0)

            signal = signal_map.get(prediction, 'N/A')

            values = (
                symbol,
                signal,
                f"{confidence*100:.1f}%",
                f"${features.get('mid_price', 0):.6f}",
                f"{features.get('spread_pct', 0):.3f}%",
                f"{features.get('volume_imbalance', 0):.2f}x",
                f"{features.get('price_momentum', 0):+.2f}%",
                '✅' if data.get('last_update') else '❌'
            )

            item = self.pairs_tree.insert('', tk.END, values=values)

            # Colora in base al segnale
            if signal == 'BUY' and confidence >= self.config['min_confidence']:
                self.pairs_tree.item(item, tags=('buy',))
            elif signal == 'SELL' and confidence >= self.config['min_confidence']:
                self.pairs_tree.item(item, tags=('sell',))

        # Tags styling
        self.pairs_tree.tag_configure('buy', background='#1b5e20')
        self.pairs_tree.tag_configure('sell', background='#b71c1c')

    def update_analytics(self):
        """Aggiorna grafici analytics"""

        # Clear axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')

        pairs_data = list(self.monitor.pairs_data.values())

        if not pairs_data:
            return

        # 1. Signal distribution
        predictions = [d.get('prediction', 1) for d in pairs_data if d.get('prediction') is not None]
        if predictions:
            signal_counts = [predictions.count(0), predictions.count(1), predictions.count(2)]
            self.ax1.bar(['SELL', 'HOLD', 'BUY'], signal_counts, color=['red', 'gray', 'green'])
            self.ax1.set_title('Signal Distribution', color='white')
            self.ax1.set_ylabel('Count', color='white')

        # 2. Confidence distribution
        confidences = [d.get('confidence', 0) for d in pairs_data if d.get('confidence', 0) > 0]
        if confidences:
            self.ax2.hist(confidences, bins=20, color='cyan', alpha=0.7)
            self.ax2.set_title('Confidence Distribution', color='white')
            self.ax2.set_xlabel('Confidence', color='white')
            self.ax2.set_ylabel('Frequency', color='white')

        # 3. Volume imbalance top 20
        imbalances = [
            (symbol, d.get('features', {}).get('volume_imbalance', 0))
            for symbol, d in list(self.monitor.pairs_data.items())[:20]
        ]
        if imbalances:
            symbols = [s[:6] for s, _ in imbalances]
            values = [v for _, v in imbalances]
            colors = ['green' if v > 1 else 'red' for v in values]

            self.ax3.barh(symbols, values, color=colors)
            self.ax3.set_title('Volume Imbalance Top 20', color='white')
            self.ax3.set_xlabel('Bid/Ask Ratio', color='white')

        # 4. Model accuracy
        if self.monitor.ml_model and self.monitor.ml_model.is_trained:
            accuracy = self.monitor.ml_model.accuracy
            self.ax4.bar(['Accuracy'], [accuracy*100], color='lime')
            self.ax4.set_ylim([0, 100])
            self.ax4.set_title('Model Performance', color='white')
            self.ax4.set_ylabel('Accuracy %', color='white')
            self.ax4.text(0, accuracy*100 + 2, f'{accuracy*100:.2f}%', ha='center', color='white', fontsize=12)

        self.fig.tight_layout()
        self.canvas.draw()

    def run(self):
        """Avvia GUI"""
        self.root.mainloop()


# === MAIN ===
if __name__ == "__main__":
    print("=" * 80)
    print("🤖 AI MULTI-PAIR MONITOR - Binance Trading Intelligence")
    print("=" * 80)
    print()
    print("📊 Features:")
    print("   ✅ Monitoraggio automatico multi-coppia (BTC/USDC/USDT)")
    print("   ✅ Machine Learning con Random Forest/Gradient Boosting")
    print("   ✅ 20+ features estratte dall'orderbook")
    print("   ✅ Segnali BUY/SELL/HOLD con confidence")
    print("   ✅ Dashboard real-time con analytics")
    print("   ✅ Auto-training e model persistence")
    print()
    print("💡 Setup:")
    print("   1. Train il modello ML (bottone 'Train Model')")
    print("   2. Salva il modello (bottone 'Save Model')")
    print("   3. Refresh pairs (bottone 'Refresh Pairs')")
    print("   4. Start monitoring (bottone 'Start Monitoring')")
    print()
    print("🚀 Avvio GUI...")
    print("=" * 80)
    print()

    try:
        app = AIMultiPairGUI()
        app.run()
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

    input("\nPremi INVIO per uscire...")
