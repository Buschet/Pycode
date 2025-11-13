#!/usr/bin/env python3
"""
Order Book Levels Analyzer - Limit Orders Strategy
Analizza il book per individuare livelli chiave e piazzare ordini limite
ENHANCED: Max 5 ordini pendenti + rimozione automatica ordini toccati
"""

import sys
sys.path.append(r'C:\Users\Utente\Desktop\Ambiente\env\Lib\site-packages')

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import requests
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional
import math

# Importa il sistema Binance esistente
try:
    from advanced_binance_integration import BinanceTradingIntegration, AdvancedBinanceAPI
except ImportError:
    print("⚠️ advanced_binance_integration.py non trovato - modalità demo")
    BinanceTradingIntegration = None

plt.style.use('dark_background')



def get_timestamp():
    """Timestamp sincronizzato con Binance"""
    try:
        # Ottieni server time da Binance
        response = requests.get('https://api.binance.com/api/v3/time', timeout=3)
        if response.status_code == 200:
            server_time = response.json()['serverTime']
            return server_time - 100  # -100ms buffer di sicurezza
    except:
        pass
    
    # Fallback: local time con correzione
    return int(time.time() * 1000) - 1000  # -1 secondo di sicurezza

@dataclass
class TradingSignal:
    """Segnale di trading"""
    timestamp: int
    symbol: str
    action: str  # BUY, SELL, HOLD
    strategy: str
    price: float
    confidence: float
    reason: str
    stop_loss: float
    take_profit: float
    position_size: float

@dataclass
class OrderBookLevel:
    """Livello significativo nel book"""
    price: float
    volume: float
    side: str  # 'bid' or 'ask'
    strength: float  # Forza del livello (1-10)
    level_type: str  # 'support', 'resistance', 'wall', 'gap'
    confidence: float
    distance_pct: float  # Distanza dal prezzo corrente in %

@dataclass
class LimitOrder:
    """Ordine limite da piazzare"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    order_type: str  # 'LIMIT', 'STOP_LIMIT'
    level_type: str
    confidence: float
    status: str = 'PENDING'
    order_id: str = None
    created_at: datetime = None
    touched_at: datetime = None  # ✅ NUOVO: timestamp quando il prezzo ha toccato l'ordine

class OrderManager:
    """✅ NUOVO: Gestore ordini con logica di rimozione automatica"""
    
    @staticmethod
    def check_orders_touched(orders, current_price, price_tolerance=0.001):
        """
        Controlla se gli ordini sono stati toccati dal prezzo
        
        Args:
            orders: Lista ordini da controllare
            current_price: Prezzo corrente
            price_tolerance: Tolleranza percentuale (0.001 = 0.1%)
        
        Returns:
            List of touched orders
        """
        touched_orders = []
        
        for order in orders:
            if order.status != 'PENDING':
                continue
            
            # Calcola la distanza percentuale dal prezzo dell'ordine
            price_diff_pct = abs(current_price - order.price) / order.price
            
            # Logica di "tocco" basata sul lato dell'ordine
            is_touched = False
            
            if order.side == 'BUY':
                # BUY order viene "toccato" se il prezzo scende al livello o sotto
                if current_price <= order.price * (1 + price_tolerance):
                    is_touched = True
                    
            elif order.side == 'SELL':
                # SELL order viene "toccato" se il prezzo sale al livello o sopra
                if current_price >= order.price * (1 - price_tolerance):
                    is_touched = True
            
            if is_touched:
                order.touched_at = datetime.now()
                order.status = 'TOUCHED'
                touched_orders.append(order)
                print(f"🎯 Order TOUCHED: {order.side} ${order.price:.2f} (current: ${current_price:.2f})")
        
        return touched_orders
    
    @staticmethod
    def remove_touched_orders(orders):
        """Rimuove ordini che sono stati toccati dal prezzo"""
        before_count = len(orders)
        orders[:] = [order for order in orders if order.status != 'TOUCHED']
        removed_count = before_count - len(orders)
        
        if removed_count > 0:
            print(f"🗑️ Removed {removed_count} touched orders")
        
        return removed_count
    
    @staticmethod
    def limit_pending_orders(orders, max_orders=5):
        """
        Limita il numero di ordini pendenti al massimo specificato
        Rimuove gli ordini meno recenti o con confidence più bassa
        """
        pending_orders = [order for order in orders if order.status == 'PENDING']
        
        if len(pending_orders) <= max_orders:
            return 0  # Nessuna rimozione necessaria
        
        # Ordina per confidence decrescente, poi per data di creazione
        pending_orders.sort(key=lambda x: (x.confidence, x.created_at or datetime.min), reverse=True)
        
        # Mantieni solo i migliori max_orders
        orders_to_keep = pending_orders[:max_orders]
        orders_to_remove = pending_orders[max_orders:]
        
        # Rimuovi dalla lista originale
        for order in orders_to_remove:
            if order in orders:
                orders.remove(order)
        
        removed_count = len(orders_to_remove)
        if removed_count > 0:
            print(f"📊 Limited to {max_orders} pending orders, removed {removed_count} lower priority orders")
        
        return removed_count
    
    @staticmethod
    def get_order_statistics(orders):
        """Ottieni statistiche degli ordini"""
        stats = {
            'total': len(orders),
            'pending': len([o for o in orders if o.status == 'PENDING']),
            'touched': len([o for o in orders if o.status == 'TOUCHED']),
            'placed': len([o for o in orders if o.status == 'PLACED']),
            'error': len([o for o in orders if o.status == 'ERROR']),
            'buy_orders': len([o for o in orders if o.side == 'BUY']),
            'sell_orders': len([o for o in orders if o.side == 'SELL'])
        }
        return stats

class OrderBookLevelsAnalyzer:
    """Analizzatore livelli Order Book"""
    
    @staticmethod
    def analyze_orderbook_levels(orderbook, current_price, config):
        """Analizza il book per trovare livelli significativi"""
        levels = []
        
        if not orderbook['bids'] or not orderbook['asks']:
            return levels
        
        # Parametri configurabili
        min_volume_threshold = config.get('min_volume_threshold', 5.0)
        max_distance_pct = config.get('max_distance_pct', 2.0)  # 2% dal prezzo corrente
        wall_multiplier = config.get('wall_multiplier', 3.0)
        
        # Analizza BID levels (support)
        avg_bid_volume = np.mean([vol for _, vol in orderbook['bids'][:20]])
        
        for price, volume in orderbook['bids'][:50]:
            distance_pct = abs((current_price - price) / current_price * 100)
            
            if distance_pct <= max_distance_pct and volume >= min_volume_threshold:
                # Calcola forza del livello
                volume_strength = min(volume / avg_bid_volume, 10.0)
                distance_strength = max(0, (max_distance_pct - distance_pct) / max_distance_pct * 5)
                strength = (volume_strength + distance_strength) / 2
                
                # Determina tipo di livello
                level_type = 'support'
                if volume >= avg_bid_volume * wall_multiplier:
                    level_type = 'wall'
                
                # Confidence basata su forza
                confidence = min(strength / 10 * 0.8 + 0.2, 1.0)
                
                levels.append(OrderBookLevel(
                    price=price,
                    volume=volume,
                    side='bid',
                    strength=strength,
                    level_type=level_type,
                    confidence=confidence,
                    distance_pct=distance_pct
                ))
        
        # Analizza ASK levels (resistance)
        avg_ask_volume = np.mean([vol for _, vol in orderbook['asks'][:20]])
        
        for price, volume in orderbook['asks'][:50]:
            distance_pct = abs((price - current_price) / current_price * 100)
            
            if distance_pct <= max_distance_pct and volume >= min_volume_threshold:
                # Calcola forza del livello
                volume_strength = min(volume / avg_ask_volume, 10.0)
                distance_strength = max(0, (max_distance_pct - distance_pct) / max_distance_pct * 5)
                strength = (volume_strength + distance_strength) / 2
                
                # Determina tipo di livello
                level_type = 'resistance'
                if volume >= avg_ask_volume * wall_multiplier:
                    level_type = 'wall'
                
                # Confidence basata su forza
                confidence = min(strength / 10 * 0.8 + 0.2, 1.0)
                
                levels.append(OrderBookLevel(
                    price=price,
                    volume=volume,
                    side='ask',
                    strength=strength,
                    level_type=level_type,
                    confidence=confidence,
                    distance_pct=distance_pct
                ))
        
        # Ordina per forza decrescente
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        # Trova GAP (zone vuote nel book)
        levels.extend(OrderBookLevelsAnalyzer.find_gaps(orderbook, current_price, config))
        
        return levels[:20]  # Top 20 livelli
    
    @staticmethod
    def find_gaps(orderbook, current_price, config):
        """Trova gap significativi nel book"""
        gaps = []
        
        min_gap_size = config.get('min_gap_pct', 0.1)  # 0.1%
        
        # Gap nei bid
        prev_price = None
        for price, volume in orderbook['bids'][:30]:
            if prev_price:
                gap_pct = (prev_price - price) / price * 100
                if gap_pct > min_gap_size:
                    gap_price = (prev_price + price) / 2
                    distance_pct = abs((current_price - gap_price) / current_price * 100)
                    
                    if distance_pct <= 2.0:  # Gap entro 2%
                        gaps.append(OrderBookLevel(
                            price=gap_price,
                            volume=0,
                            side='bid',
                            strength=gap_pct * 2,  # Forza basata su dimensione gap
                            level_type='gap',
                            confidence=0.6,
                            distance_pct=distance_pct
                        ))
            prev_price = price
        
        # Gap negli ask
        prev_price = None
        for price, volume in orderbook['asks'][:30]:
            if prev_price:
                gap_pct = (price - prev_price) / prev_price * 100
                if gap_pct > min_gap_size:
                    gap_price = (prev_price + price) / 2
                    distance_pct = abs((gap_price - current_price) / current_price * 100)
                    
                    if distance_pct <= 2.0:  # Gap entro 2%
                        gaps.append(OrderBookLevel(
                            price=gap_price,
                            volume=0,
                            side='ask',
                            strength=gap_pct * 2,
                            level_type='gap',
                            confidence=0.6,
                            distance_pct=distance_pct
                        ))
            prev_price = price
        
        return gaps

class LimitOrderStrategy:
    """Strategia per piazzare ordini limite sui livelli"""
    
    @staticmethod
    def generate_limit_orders(levels, current_price, config, max_orders=5):
        """
        Genera ordini limite basati sui livelli trovati - ✅ ENHANCED con limite ordini
        
        Args:
            levels: Livelli rilevati
            current_price: Prezzo corrente
            config: Configurazione
            max_orders: Numero massimo di ordini da generare
        """
        orders = []
        
        symbol = config.get('symbol', 'BTCUSDC')
        base_quantity = config.get('base_quantity', 0.001)
        min_confidence = config.get('min_confidence_limits', 0.7)
        
        print(f"🔧 Order generation config (MAX {max_orders} orders):")
        print(f"   Symbol: {symbol}")
        print(f"   Base quantity: {base_quantity}")
        print(f"   Min confidence: {min_confidence}")
        print(f"   Current price: ${current_price:.2f}")
        
        orders_created = 0
        
        # ✅ Ordina i livelli per priorità: confidence * strength
        levels_prioritized = sorted(levels, key=lambda x: x.confidence * x.strength, reverse=True)
        
        for level in levels_prioritized:
            # ✅ STOP se abbiamo raggiunto il limite
            if orders_created >= max_orders:
                print(f"🛑 Max orders limit reached ({max_orders})")
                break
                
            print(f"📊 Processing level: ${level.price:.2f} | {level.side} | {level.level_type} | conf:{level.confidence:.2f}")
            
            if level.confidence < min_confidence:
                print(f"   ❌ Skipped: confidence {level.confidence:.2f} < {min_confidence}")
                continue
            
            # ✅ STRATEGIA BUY LIMIT sui supporti (BID levels)
            if level.side == 'bid' and level.level_type in ['support', 'wall']:
                # BUY LIMIT: piazza ESATTAMENTE al livello supporto (aspetta che il prezzo scenda)
                buy_price = level.price  # Nessun aggiustamento - usa il supporto esatto
                
                print(f"   🟢 BUY LIMIT: level=${level.price:.2f}, order_price=${buy_price:.2f}, current=${current_price:.2f}")
                
                if buy_price < current_price:  # Solo se sotto prezzo corrente
                    quantity = base_quantity * max(1.0, level.strength / 5)
                    
                    orders.append(LimitOrder(
                        symbol=symbol,
                        side='BUY',
                        price=buy_price,
                        quantity=quantity,
                        order_type='LIMIT',
                        level_type=f'{level.level_type}_support',
                        confidence=level.confidence,
                        created_at=datetime.now()
                    ))
                    
                    orders_created += 1
                    print(f"   ✅ BUY LIMIT created: ${buy_price:.2f} x {quantity:.4f} (Support Level)")
                else:
                    print(f"   ⚠️ BUY LIMIT skipped: level ${buy_price:.2f} >= current ${current_price:.2f}")
            
            # ✅ STRATEGIA SELL LIMIT sulle resistenze (ASK levels)
            elif level.side == 'ask' and level.level_type in ['resistance', 'wall']:
                # SELL LIMIT: piazza ESATTAMENTE al livello resistenza (aspetta che il prezzo salga)
                sell_price = level.price  # Nessun aggiustamento - usa la resistenza esatta
                
                print(f"   🔴 SELL LIMIT: level=${level.price:.2f}, order_price=${sell_price:.2f}, current=${current_price:.2f}")
                
                if sell_price > current_price:  # Solo se sopra prezzo corrente
                    quantity = base_quantity * max(1.0, level.strength / 5)
                    
                    orders.append(LimitOrder(
                        symbol=symbol,
                        side='SELL',
                        price=sell_price,
                        quantity=quantity,
                        order_type='LIMIT',
                        level_type=f'{level.level_type}_resistance',
                        confidence=level.confidence,
                        created_at=datetime.now()
                    ))
                    
                    orders_created += 1
                    print(f"   ✅ SELL LIMIT created: ${sell_price:.2f} x {quantity:.4f} (Resistance Level)")
                else:
                    print(f"   ⚠️ SELL LIMIT skipped: level ${sell_price:.2f} <= current ${current_price:.2f}")
            
            # ✅ STRATEGIA GAP - Piazza ordini nei gap per catturare movimenti
            elif level.level_type == 'gap' and orders_created < max_orders:
                print(f"   🕳️ GAP STRATEGY: level=${level.price:.2f}, side={level.side}")
                
                if level.side == 'bid' and level.price < current_price * 0.99:  # Gap nei bid sotto prezzo
                    # Piazza BUY nel gap per catturare dip
                    quantity = base_quantity * 1.5
                    
                    orders.append(LimitOrder(
                        symbol=symbol,
                        side='BUY',
                        price=level.price,
                        quantity=quantity,
                        order_type='LIMIT',
                        level_type='gap_buy',
                        confidence=level.confidence,
                        created_at=datetime.now()
                    ))
                    
                    orders_created += 1
                    print(f"   ✅ GAP BUY created: ${level.price:.2f} x {quantity:.4f}")
                    
                elif level.side == 'ask' and level.price > current_price * 1.01:  # Gap negli ask sopra prezzo
                    # Piazza SELL nel gap per catturare pump
                    quantity = base_quantity * 1.5
                    
                    orders.append(LimitOrder(
                        symbol=symbol,
                        side='SELL',
                        price=level.price,
                        quantity=quantity,
                        order_type='LIMIT',
                        level_type='gap_sell',
                        confidence=level.confidence,
                        created_at=datetime.now()
                    ))
                    
                    orders_created += 1
                    print(f"   ✅ GAP SELL created: ${level.price:.2f} x {quantity:.4f}")
                else:
                    print(f"   ⚠️ GAP skipped: not suitable distance from current price")
            else:
                print(f"   ⏭️ Skipped: unsupported level type, side, or not suitable for current strategy")
        
        print(f"🎯 Total orders created: {orders_created} (limit: {max_orders})")
        
        # Ordina per confidence decrescente
        orders.sort(key=lambda x: x.confidence, reverse=True)
        
        return orders

class OrderBookLevelsGUI:
    """GUI per Order Book Levels Analyzer"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Order Book Levels Analyzer - Enhanced Order Management")
        self.root.geometry("1500x950")
        self.root.configure(bg='#1a1a1a')
        
        # Stato sistema
        self.is_running = False
        self.current_symbol = tk.StringVar(value="BTCUSDC")
        
        # Dati
        self.orderbook = {'bids': [], 'asks': []}
        self.current_levels = []
        self.pending_orders = []
        self.executed_orders = []
        self.price_history = deque(maxlen=100)
        
        # ✅ NEW: Order management settings
        self.max_pending_orders = 5
        self.price_tolerance = 0.002  # 0.2% tolerance for order touching
        
        # Configurazione DEFAULT PIÙ PERMISSIVA
        self.config = {
            'symbol': 'BTCUSDC',
            'min_volume_threshold': 1.0,      # ✅ Ridotto da 5.0 a 1.0
            'max_distance_pct': 5.0,          # ✅ Aumentato da 2.0 a 5.0
            'wall_multiplier': 1.5,           # ✅ Ridotto da 3.0 a 1.5
            'min_gap_pct': 0.05,              # ✅ Ridotto da 0.1 a 0.05
            'base_quantity': 0.001,
            'min_confidence_limits': 0.10,    # ✅ Ridotto da 0.75 a 0.10
            'update_interval': 10,
            'auto_place_orders': False,
            'max_pending_orders': 5,          # ✅ NEW
            'price_tolerance': 0.002          # ✅ NEW
        }
        
        # Simboli disponibili
        self.symbols = ['BTCUSDC', 'ETHUSDC', 'ADAUSDT', 'SOLUSDT']
        
        # Setup Binance se disponibile
        self.setup_binance()
        
        self.setup_gui()
        
    def setup_binance(self):
        """Setup integrazione Binance - ENHANCED DIAGNOSTIC WITH DETAILED ERROR CHECKING"""
        print("\n" + "="*60)
        print("🔧 BINANCE INTEGRATION DIAGNOSTIC")
        print("="*60)
        
        if BinanceTradingIntegration:
            try:
                print("📋 Step 1: Loading Binance Integration...")
                API_KEY = "seU99BIqWSVbtZ8PmW0PTnNSLpWsj8WE43JFKwzLHPGu7Wb4ZFwE6fjddljcGK87"
                API_SECRET = "0snc4bvVMlK0OlSahiW0grsMrRAzapDj17J99gGxokes1LRZyi2NEs9n4vMJ6iVx"
                            
                print(f"   API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
                print(f"   API Secret: {API_SECRET[:10]}...***")
                
                self.binance = BinanceTradingIntegration(API_KEY, API_SECRET, testnet=False)
                print("✅ Binance integration object created successfully")
                
                print("\n📋 Step 2: Analyzing Available Methods...")
                
                # Get all available attributes
                all_attrs = dir(self.binance)
                methods = [attr for attr in all_attrs if not attr.startswith('_')]
                print(f"   Total available attributes: {len(methods)}")
                print(f"   Methods: {methods[:10]}...")  # Show first 10
                
                # Check for common trading methods
                trading_methods = {
                    'place_order': hasattr(self.binance, 'place_order'),
                    'place_limit_order': hasattr(self.binance, 'place_limit_order'),
                    'cancel_order': hasattr(self.binance, 'cancel_order'),
                    'get_account': hasattr(self.binance, 'get_account'),
                    'get_account_info': hasattr(self.binance, 'get_account_info'),
                    'client': hasattr(self.binance, 'client')
                }
                
                print("   Trading methods check:")
                for method, available in trading_methods.items():
                    status = "✅" if available else "❌"
                    print(f"     {status} {method}: {available}")
                
                print("\n📋 Step 3: Checking Client Access...")
                
                # Check client if available
                if hasattr(self.binance, 'client'):
                    client = self.binance.client
                    print("✅ Client object found")
                    
                    # Check client type
                    client_type = type(client).__name__
                    print(f"   Client type: {client_type}")
                    
                    # Check client methods
                    client_methods = [attr for attr in dir(client) if not attr.startswith('_')]
                    order_methods = [m for m in client_methods if 'order' in m.lower()]
                    account_methods = [m for m in client_methods if 'account' in m.lower()]
                    
                    print(f"   Client order methods: {order_methods}")
                    print(f"   Client account methods: {account_methods}")
                    
                    # Test basic connectivity
                    try:
                        print("\n📋 Step 4: Testing Basic Connectivity...")
                        
                        # Try ping first
                        if hasattr(client, 'ping'):
                            ping_result = client.ping()
                            print(f"✅ Ping successful: {ping_result}")
                        
                        # Try server time
                        if hasattr(client, 'get_server_time'):
                            server_time = client.get_server_time()
                            print(f"✅ Server time: {server_time}")
                        
                        # Try exchange info
                        if hasattr(client, 'get_exchange_info'):
                            exchange_info = client.get_exchange_info()
                            print(f"✅ Exchange info received (symbols: {len(exchange_info.get('symbols', []))})")
                        
                    except Exception as connectivity_error:
                        print(f"⚠️ Basic connectivity test failed: {connectivity_error}")
                    
                    # Test account access
                    try:
                        print("\n📋 Step 5: Testing Account Access...")
                        
                        if hasattr(client, 'get_account'):
                            account_info = client.get_account()
                            print("✅ Account info retrieved successfully")
                            print(f"   Account type: {account_info.get('accountType', 'Unknown')}")
                            print(f"   Can trade: {account_info.get('canTrade', False)}")
                            print(f"   Balances count: {len(account_info.get('balances', []))}")
                            
                            # Show some balances
                            balances = account_info.get('balances', [])
                            non_zero_balances = [b for b in balances if float(b.get('free', 0)) > 0]
                            print(f"   Non-zero balances: {len(non_zero_balances)}")
                            for balance in non_zero_balances[:5]:  # Show first 5
                                asset = balance.get('asset', 'Unknown')
                                free = balance.get('free', '0')
                                print(f"     {asset}: {free}")
                                
                        elif hasattr(client, 'get_account_status'):
                            status = client.get_account_status()
                            print(f"✅ Account status: {status}")
                            
                    except Exception as account_error:
                        print(f"❌ Account access failed: {account_error}")
                        print(f"   Error type: {type(account_error).__name__}")
                        
                        # Check specific error types
                        error_str = str(account_error).lower()
                        if 'api' in error_str and 'key' in error_str:
                            print("   🔑 Possible API key issue")
                        elif 'signature' in error_str:
                            print("   ✍️ Possible signature/secret issue")
                        elif 'permission' in error_str:
                            print("   🚫 Possible permission issue")
                        elif 'timestamp' in error_str:
                            print("   ⏰ Possible timestamp issue")
                
                else:
                    print("❌ No client object found")
                
                print("\n📋 Step 6: Symbol Validation...")
                
                # Test symbol validation
                test_symbols = ['BTCUSDT', 'BTCUSDC', 'ETHUSDT', 'ETHUSDC']
                try:
                    if hasattr(self.binance, 'client') and hasattr(self.binance.client, 'get_symbol_info'):
                        for symbol in test_symbols:
                            try:
                                symbol_info = self.binance.client.get_symbol_info(symbol)
                                if symbol_info:
                                    print(f"   ✅ {symbol}: Valid")
                                else:
                                    print(f"   ❌ {symbol}: Not found")
                            except:
                                print(f"   ⚠️ {symbol}: Error checking")
                    else:
                        print("   ⚠️ Cannot validate symbols - no symbol info method")
                except Exception as symbol_error:
                    print(f"   ❌ Symbol validation error: {symbol_error}")
                
                print("\n" + "="*60)
                print("✅ BINANCE DIAGNOSTIC COMPLETE")
                print("="*60)
                
            except Exception as setup_error:
                print(f"❌ CRITICAL: Binance setup failed: {setup_error}")
                print(f"   Error type: {type(setup_error).__name__}")
                import traceback
                print("   Full traceback:")
                traceback.print_exc()
                self.binance = None
        else:
            self.binance = None
            print("❌ BinanceTradingIntegration class not available")
            print("💡 Ensure advanced_binance_integration.py is present and working")
            print("📁 Expected file: advanced_binance_integration.py")
            print("🔧 Should contain: class BinanceTradingIntegration")
    
    def setup_gui(self):
        """Setup interfaccia principale"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.setup_header(main_frame)
        
        # Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Tabs
        self.setup_levels_tab()
        self.setup_orders_tab()
        self.setup_config_tab()
        
    def setup_header(self, parent):
        """Header controlli"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Simbolo
        symbol_frame = ttk.LabelFrame(header_frame, text="📊 Symbol", padding=10)
        symbol_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.current_symbol,
                                        values=self.symbols, width=15, state='readonly')
        self.symbol_combo.pack()
        
        # Controlli
        controls_frame = ttk.LabelFrame(header_frame, text="🎮 Controls", padding=10)
        controls_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.start_btn = ttk.Button(controls_frame, text="🚀 START ANALYSIS",
                                   command=self.start_analysis, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(controls_frame, text="⏹️ STOP",
                                  command=self.stop_analysis, width=15, state='disabled')
        self.stop_btn.pack(side=tk.LEFT)
        
        # Status ENHANCED
        status_frame = ttk.LabelFrame(header_frame, text="📊 Enhanced Status", padding=10)
        status_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(status_frame, text="Sistema: Fermo", foreground='red')
        self.status_label.pack()
        
        self.levels_count_label = ttk.Label(status_frame, text="Livelli: 0")
        self.levels_count_label.pack()
        
        self.orders_count_label = ttk.Label(status_frame, text="Ordini: 0/5", foreground='orange')  # ✅ NEW
        self.orders_count_label.pack()
        
        self.touched_count_label = ttk.Label(status_frame, text="Touched: 0", foreground='yellow')  # ✅ NEW
        self.touched_count_label.pack()
    
    def setup_levels_tab(self):
        """Tab analisi livelli"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Book Levels")
        
        # Grafici frame
        charts_frame = ttk.Frame(frame)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup grafici
        self.fig_levels = plt.Figure(figsize=(14, 8), facecolor='#1a1a1a')
        
        # 2x1 grid
        self.ax_book = self.fig_levels.add_subplot(121)
        self.ax_levels = self.fig_levels.add_subplot(122)
        
        self.ax_book.set_title('Order Book + Levels', color='white', fontsize=12)
        self.ax_levels.set_title('Levels Analysis', color='white', fontsize=12)
        
        for ax in [self.ax_book, self.ax_levels]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.3)
        
        self.fig_levels.tight_layout(pad=3.0)
        
        self.canvas_levels = FigureCanvasTkAgg(self.fig_levels, charts_frame)
        self.canvas_levels.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Levels table
        levels_table_frame = ttk.LabelFrame(frame, text="🎯 Detected Levels", padding=10)
        levels_table_frame.pack(fill=tk.X, pady=(10, 0))
        
        columns = ('Price', 'Volume', 'Side', 'Type', 'Strength', 'Distance', 'Confidence')
        self.levels_tree = ttk.Treeview(levels_table_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.levels_tree.heading(col, text=col)
            self.levels_tree.column(col, width=100)
        
        scrollbar_levels = ttk.Scrollbar(levels_table_frame, orient=tk.VERTICAL, command=self.levels_tree.yview)
        self.levels_tree.configure(yscrollcommand=scrollbar_levels.set)
        
        self.levels_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_levels.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_orders_tab(self):
        """Tab gestione ordini - ENHANCED"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋 Enhanced Orders")
        
        # Orders table ENHANCED
        orders_frame = ttk.LabelFrame(frame, text="🎯 Generated Limit Orders (Max 5 Pending)", padding=10)
        orders_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Side', 'Price', 'Quantity', 'Type', 'Level Type', 'Confidence', 'Status', 'Created', 'Touched')  # ✅ NEW columns
        self.orders_tree = ttk.Treeview(orders_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.orders_tree.heading(col, text=col)
            if col in ['Created', 'Touched']:
                self.orders_tree.column(col, width=80)
            else:
                self.orders_tree.column(col, width=100)
        
        scrollbar_orders = ttk.Scrollbar(orders_frame, orient=tk.VERTICAL, command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=scrollbar_orders.set)
        
        self.orders_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_orders.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Order controls ENHANCED
        controls_frame = ttk.Frame(orders_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Row 1
        row1 = ttk.Frame(controls_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row1, text="📋 Generate Orders", 
                  command=self.generate_orders_manual).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row1, text="🔄 Clear All Orders", 
                  command=self.clear_orders).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row1, text="🗑️ Remove Touched", 
                  command=self.remove_touched_orders_manual).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row1, text="🎯 Check Touch", 
                  command=self.check_orders_touched_manual).pack(side=tk.LEFT, padx=(0, 10))
        
        # Row 2
        row2 = ttk.Frame(controls_frame)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row2, text="🚀 Place Selected", 
                  command=self.place_selected_orders).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row2, text="❌ Cancel All", 
                  command=self.cancel_all_orders).pack(side=tk.LEFT, padx=(0, 10))
        
        # Max orders control
        max_orders_frame = ttk.Frame(row2)
        max_orders_frame.pack(side=tk.RIGHT)
        
        ttk.Label(max_orders_frame, text="Max Orders:").pack(side=tk.LEFT)
        self.max_orders_var = tk.IntVar(value=5)
        ttk.Spinbox(max_orders_frame, from_=1, to=20, width=5, 
                   textvariable=self.max_orders_var, 
                   command=self.update_max_orders).pack(side=tk.LEFT, padx=(5, 0))
        
        # Statistics frame ✅ NEW
        stats_frame = ttk.LabelFrame(orders_frame, text="📊 Order Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X)
        
        self.stats_label = ttk.Label(stats_inner, text="Ready to analyze", font=('Arial', 9))
        self.stats_label.pack(side=tk.LEFT)
        
        # Price monitoring ✅ NEW
        price_frame = ttk.Frame(stats_inner)
        price_frame.pack(side=tk.RIGHT)
        
        ttk.Label(price_frame, text="Price Tolerance:").pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.002)
        ttk.Spinbox(price_frame, from_=0.001, to=0.010, increment=0.001, width=8,
                   textvariable=self.tolerance_var, format="%.3f").pack(side=tk.LEFT, padx=(5, 0))
        
        # Debug info ENHANCED with Binance status
        debug_frame = ttk.LabelFrame(orders_frame, text="🔍 Enhanced Debug Info", padding=5)
        debug_frame.pack(fill=tk.X, pady=(10, 0))
        
        debug_inner = ttk.Frame(debug_frame)
        debug_inner.pack(fill=tk.X)
        
        self.debug_label = ttk.Label(debug_inner, text="Status: Ready for enhanced order management", font=('Arial', 8))
        self.debug_label.pack(side=tk.LEFT)
        
        # ✅ NEW: Binance connection status
        self.binance_status_label = ttk.Label(debug_inner, 
            text="Binance: Checking...", 
            font=('Arial', 8), 
            foreground='orange')
        self.binance_status_label.pack(side=tk.RIGHT)
        
        # Update Binance status
        self.root.after(1000, self.update_binance_status)  # Check after 1 second

    def setup_config_tab(self):
        """Tab configurazione - ENHANCED"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ Enhanced Config")
        
        config_frame = ttk.LabelFrame(frame, text="🔧 Analysis Parameters", padding=20)
        config_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Min Volume Threshold
        ttk.Label(config_frame, text="Min Volume Threshold:").grid(row=0, column=0, sticky='w', pady=5)
        self.min_volume_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(config_frame, from_=0.1, to=100.0, increment=0.1,
                   textvariable=self.min_volume_var, width=15, format="%.1f").grid(row=0, column=1, pady=5, sticky='w')
        
        # Max Distance %
        ttk.Label(config_frame, text="Max Distance (%):").grid(row=1, column=0, sticky='w', pady=5)
        self.max_distance_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(config_frame, from_=0.1, to=20.0, increment=0.1,
                   textvariable=self.max_distance_var, width=15, format="%.1f").grid(row=1, column=1, pady=5, sticky='w')
        
        # Wall Multiplier
        ttk.Label(config_frame, text="Wall Multiplier:").grid(row=2, column=0, sticky='w', pady=5)
        self.wall_multiplier_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(config_frame, from_=0.5, to=20.0, increment=0.1,
                   textvariable=self.wall_multiplier_var, width=15, format="%.1f").grid(row=2, column=1, pady=5, sticky='w')
        
        # Base Quantity
        ttk.Label(config_frame, text="Base Quantity:").grid(row=3, column=0, sticky='w', pady=5)
        self.base_quantity_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(config_frame, from_=0.00001, to=1.0, increment=0.00001,
                   textvariable=self.base_quantity_var, width=15, format="%.5f").grid(row=3, column=1, pady=5, sticky='w')
        
        # Min Confidence
        ttk.Label(config_frame, text="Min Confidence:").grid(row=4, column=0, sticky='w', pady=5)
        self.min_confidence_var = tk.DoubleVar(value=0.10)
        ttk.Spinbox(config_frame, from_=0.01, to=0.99, increment=0.01,
                   textvariable=self.min_confidence_var, width=15, format="%.2f").grid(row=4, column=1, pady=5, sticky='w')
        
        # Update Interval
        ttk.Label(config_frame, text="Update Interval (sec):").grid(row=5, column=0, sticky='w', pady=5)
        self.update_interval_var = tk.IntVar(value=10)
        ttk.Spinbox(config_frame, from_=1, to=3600, increment=1,
                   textvariable=self.update_interval_var, width=15).grid(row=5, column=1, pady=5, sticky='w')
        
        # Min Gap %
        ttk.Label(config_frame, text="Min Gap (%):").grid(row=6, column=0, sticky='w', pady=5)
        self.min_gap_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(config_frame, from_=0.01, to=2.0, increment=0.01,
                   textvariable=self.min_gap_var, width=15, format="%.2f").grid(row=6, column=1, pady=5, sticky='w')
        
        # ✅ NEW: Enhanced Order Management Settings
        order_mgmt_frame = ttk.LabelFrame(config_frame, text="🎯 Enhanced Order Management", padding=10)
        order_mgmt_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky='ew')
        
        # Max pending orders
        ttk.Label(order_mgmt_frame, text="Max Pending Orders:").grid(row=0, column=0, sticky='w', pady=2)
        self.max_pending_var = tk.IntVar(value=5)
        ttk.Spinbox(order_mgmt_frame, from_=1, to=50, increment=1,
                   textvariable=self.max_pending_var, width=10).grid(row=0, column=1, pady=2, sticky='w')
        
        # Price tolerance
        ttk.Label(order_mgmt_frame, text="Price Touch Tolerance (%):").grid(row=1, column=0, sticky='w', pady=2)
        self.price_tolerance_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(order_mgmt_frame, from_=0.01, to=2.0, increment=0.01,
                   textvariable=self.price_tolerance_var, width=10, format="%.2f").grid(row=1, column=1, pady=2, sticky='w')
        
        # Auto remove touched orders
        self.auto_remove_touched_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(order_mgmt_frame, text="🗑️ Auto Remove Touched Orders",
                       variable=self.auto_remove_touched_var).grid(row=2, column=0, columnspan=2, sticky='w', pady=2)
        
        # Auto limit pending orders
        self.auto_limit_pending_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(order_mgmt_frame, text="📊 Auto Limit Pending Orders",
                       variable=self.auto_limit_pending_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=2)
        
        # Auto Generate Orders
        auto_generate_frame = ttk.LabelFrame(config_frame, text="🤖 Auto Order Generation", padding=10)
        auto_generate_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.auto_generate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_generate_frame, text="🔄 Auto Generate Orders from Levels",
                       variable=self.auto_generate_var).pack(side=tk.LEFT)
        
        self.auto_place_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(auto_generate_frame, text="🚀 Auto Place Orders (LIVE)",
                       variable=self.auto_place_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(config_frame, text="🎯 Preset Configurations", padding=10)
        preset_frame.grid(row=9, column=0, columnspan=2, pady=10, sticky='ew')
        
        ttk.Button(preset_frame, text="🔥 Aggressive", 
                  command=self.set_aggressive_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="⚖️ Balanced", 
                  command=self.set_balanced_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="🛡️ Conservative", 
                  command=self.set_conservative_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="🚀 No Limits", 
                  command=self.set_no_limits_preset).pack(side=tk.LEFT, padx=5)
        
        # Update Config Button
        ttk.Button(config_frame, text="🔄 Update Enhanced Configuration",
                  command=self.update_config).grid(row=10, column=0, columnspan=2, pady=20)
    
    def update_max_orders(self):
        """✅ NEW: Aggiorna il numero massimo di ordini"""
        self.max_pending_orders = self.max_orders_var.get()
        self.config['max_pending_orders'] = self.max_pending_orders
        print(f"🔄 Max pending orders updated to: {self.max_pending_orders}")
        
        # Applica limite immediato se necessario
        if len([o for o in self.pending_orders if o.status == 'PENDING']) > self.max_pending_orders:
            removed = OrderManager.limit_pending_orders(self.pending_orders, self.max_pending_orders)
            if removed > 0:
                self.update_orders_display()
                self.update_debug_info(f"Limited to {self.max_pending_orders} orders, removed {removed}")
    
    def check_orders_touched_manual(self):
        """✅ NEW: Controlla manualmente se ordini sono stati toccati"""
        if not self.pending_orders:
            messagebox.showwarning("Warning", "Nessun ordine da controllare!")
            return
        
        if not self.orderbook['bids'] or not self.orderbook['asks']:
            messagebox.showwarning("Warning", "Order book non disponibile!")
            return
        
        current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
        tolerance = self.tolerance_var.get()
        
        touched_orders = OrderManager.check_orders_touched(
            self.pending_orders, current_price, tolerance
        )
        
        if touched_orders:
            self.update_orders_display()
            messagebox.showinfo("Orders Touched", 
                f"Found {len(touched_orders)} touched orders!\n"
                f"Current price: ${current_price:.2f}\n"
                f"Tolerance: {tolerance:.3f}%")
        else:
            messagebox.showinfo("Orders Check", "No orders touched by current price.")
    
    def remove_touched_orders_manual(self):
        """✅ NEW: Rimuove manualmente ordini toccati"""
        before_count = len(self.pending_orders)
        removed_count = OrderManager.remove_touched_orders(self.pending_orders)
        
        if removed_count > 0:
            self.update_orders_display()
            self.update_order_statistics()
            messagebox.showinfo("Orders Removed", 
                f"Removed {removed_count} touched orders!\n"
                f"Remaining orders: {len(self.pending_orders)}")
        else:
            messagebox.showinfo("Orders Clean", "No touched orders to remove.")
    
    def update_order_statistics(self):
        """✅ NEW: Aggiorna statistiche ordini"""
        stats = OrderManager.get_order_statistics(self.pending_orders)
        
        stats_text = (f"Total: {stats['total']} | "
                     f"Pending: {stats['pending']}/{self.max_pending_orders} | "
                     f"Touched: {stats['touched']} | "
                     f"BUY: {stats['buy_orders']} | "
                     f"SELL: {stats['sell_orders']}")
        
        self.root.after(0, lambda: self.stats_label.config(text=stats_text))
        
        # Update header counts
        self.root.after(0, lambda: self.orders_count_label.config(
            text=f"Ordini: {stats['pending']}/{self.max_pending_orders}",
            foreground='green' if stats['pending'] <= self.max_pending_orders else 'red'
        ))
        
        self.root.after(0, lambda: self.touched_count_label.config(
            text=f"Touched: {stats['touched']}",
            foreground='yellow' if stats['touched'] > 0 else 'gray'
        ))
    
    def fetch_orderbook(self):
        """Fetch order book"""
        try:
            symbol = self.current_symbol.get()
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                bids = [[float(price), float(volume)] for price, volume in data['bids']]
                asks = [[float(price), float(volume)] for price, volume in data['asks']]
                
                self.orderbook = {'bids': bids, 'asks': asks}
                return True
            return False
        except:
            return False
    
    def analyze_levels(self):
        """Analizza livelli nel book"""
        if not self.orderbook['bids'] or not self.orderbook['asks']:
            return []
        
        current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
        self.price_history.append(current_price)
        
        levels = OrderBookLevelsAnalyzer.analyze_orderbook_levels(
            self.orderbook, current_price, self.config
        )
        
        return levels
    
    def generate_orders_manual(self):
        """Genera ordini manualmente - ENHANCED"""
        if not self.current_levels:
            messagebox.showwarning("Warning", "Nessun livello rilevato. Avvia prima l'analisi.")
            return
        
        if not self.orderbook['bids'] or not self.orderbook['asks']:
            messagebox.showwarning("Warning", "Order book non disponibile. Avvia prima l'analisi.")
            return
        
        current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
        
        print(f"🎯 Generating orders from {len(self.current_levels)} levels...")
        print(f"💰 Current price: ${current_price:.2f}")
        print(f"⚙️ Config: min_confidence={self.config.get('min_confidence_limits', 0.75)}")
        print(f"📊 Max orders: {self.max_pending_orders}")
        
        # Conta ordini pendenti attuali
        current_pending = len([o for o in self.pending_orders if o.status == 'PENDING'])
        available_slots = max(0, self.max_pending_orders - current_pending)
        
        if available_slots == 0:
            messagebox.showwarning("Warning", 
                f"Limite ordini raggiunto ({current_pending}/{self.max_pending_orders})!\n"
                f"Rimuovi ordini esistenti o aumenta il limite.")
            return
        
        print(f"📊 Current pending: {current_pending}, Available slots: {available_slots}")
        
        # Genera solo gli ordini che possiamo aggiungere
        orders = LimitOrderStrategy.generate_limit_orders(
            self.current_levels, current_price, self.config, max_orders=available_slots
        )
        
        print(f"✅ Generated {len(orders)} limit orders for {available_slots} available slots")
        
        if not orders:
            high_conf_levels = [l for l in self.current_levels if l.confidence >= self.config.get('min_confidence_limits', 0.75)]
            print(f"🔍 Debug: {len(high_conf_levels)} levels above min confidence")
            
            messagebox.showwarning("Warning", 
                f"Nessun ordine generato!\n\n"
                f"Livelli trovati: {len(self.current_levels)}\n"
                f"Confidence >= {self.config.get('min_confidence_limits', 0.75)}: {len(high_conf_levels)}\n"
                f"Slot disponibili: {available_slots}\n"
                f"Prova a ridurre la Min Confidence nel tab Configuration.")
            return
        
        # Aggiungi agli ordini esistenti
        self.pending_orders.extend(orders)
        self.update_orders_display()
        self.update_order_statistics()
        
        messagebox.showinfo("Success", 
            f"Generati {len(orders)} nuovi ordini limite!\n"
            f"Totale ordini: {len(self.pending_orders)}\n"
            f"Ordini pendenti: {len([o for o in self.pending_orders if o.status == 'PENDING'])}/{self.max_pending_orders}")
    
    def place_selected_orders(self):
        """Piazza ordini selezionati"""
        selected_items = self.orders_tree.selection()
        
        if not selected_items:
            messagebox.showwarning("Warning", "Seleziona almeno un ordine!")
            return
        
        if not self.binance:
            messagebox.showwarning("Warning", "Integrazione Binance non disponibile!")
            return
        
        placed_count = 0
        
        for item in selected_items:
            index = self.orders_tree.index(item)
            if index < len(self.pending_orders):
                order = self.pending_orders[index]
                
                if self.place_limit_order(order):
                    placed_count += 1
        
        self.update_orders_display()
        self.update_order_statistics()
        messagebox.showinfo("Success", f"Piazzati {placed_count} ordini!")

    
    def execute_signal_simulation(self, signal: TradingSignal):
        """Simulazione portfolio"""
        try:
            symbol = signal.symbol
            action = signal.action
            price = signal.price
            size = signal.position_size
            
            if action == 'BUY':
                cost = price * size * self.portfolio.cash
                if cost <= self.portfolio.cash:
                    self.portfolio.cash -= cost
                    self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + (size * self.portfolio.cash)
                    self.portfolio.total_trades += 1
                    self.log_trade(f"SIM BUY {symbol}: ${price:.6f} - Size: {size:.3f} - Cost: ${cost:.2f}")
                    
            elif action == 'SELL':
                if symbol in self.portfolio.positions and self.portfolio.positions[symbol] > 0:
                    sell_amount = min(self.portfolio.positions[symbol], size * self.portfolio.cash)
                    proceeds = (sell_amount / price) * price
                    
                    self.portfolio.cash += proceeds
                    self.portfolio.positions[symbol] -= sell_amount
                    
                    if self.portfolio.positions[symbol] <= 0:
                        del self.portfolio.positions[symbol]
                        
                    self.portfolio.total_trades += 1
                    self.log_trade(f"SIM SELL {symbol}: ${price:.6f} - Amount: ${sell_amount:.2f} - Proceeds: ${proceeds:.2f}")
                    
            self.update_portfolio_display()
            
        except Exception as e:
            self.log_trade(f"Errore simulazione: {e}")
    
    def execute_live_trade(self, signal: TradingSignal):
        """Esegue trade LIVE su Binance"""
        try:
            print(f"\n🎯 EXECUTING LIVE TRADE:")
            print(f"   Symbol: {signal.symbol}")
            print(f"   Action: {signal.action}")
            print(f"   Strategy: {signal.strategy}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Price: ${signal.price:.6f}")
            
            # Esegui su Binance
            result = self.binance_trading.execute_signal(signal)
            
            print(f"   Result: {result['status'].upper()}")
            
            self.live_trades_count += 1
            
            if result['status'] == 'executed':
                trade_info = result['result']
                order_id = trade_info.get('entry_order', {}).get('orderId', 'N/A')
                print(f"   ✅ Order ID: {order_id}")
                
                self.live_trades_success += 1
                
                # Aggiorna status GUI
                success_rate = (self.live_trades_success / self.live_trades_count) * 100
                self.update_binance_status(f"🔗 Binance: {self.live_trades_success}/{self.live_trades_count} trades ({success_rate:.1f}%)")
                
            elif result['status'] == 'skipped':
                print(f"   ⏸️ Skipped: {result['reason']}")
                
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_result = {'status': 'error', 'error': str(e)}
            print(f"   💥 Exception: {e}")
            return error_result
    
    def place_limit_order(self, order):

        signal=(TradingSignal(
            timestamp=get_timestamp(),
            symbol=order.symbol,
            action=order.side.upper(),
            strategy='Wall_Break',
            price=order.price,
            confidence=1,
            reason=f'LIMIT',
            stop_loss=order.price * 1.015,
            take_profit=order.price * 0.97,
            position_size=order.quantity
        ))

        """VERSIONE ENHANCED - Simulazione + Trading LIVE"""
        
        # 1. 📊 SIMULAZIONE (mantieni logica esistente)
        self.execute_signal_simulation(signal)
        
        # 2. 🚀 TRADING LIVE BINANCE  
        if hasattr(self, 'binance_trading') and self.binance_trading:
            live_result = self.execute_live_trade(signal)
            
            # Log risultato nel sistema
            if live_result['status'] == 'executed':
                self.log_live_trade_success(signal, live_result)
            elif live_result['status'] == 'failed':
                self.log_live_trade_error(signal, live_result)
    

    
    def update_binance_status(self):
        """✅ NEW: Aggiorna status connessione Binance"""
        if self.binance:
            try:
                # Test basic connection
                if hasattr(self.binance, 'get_account_info'):
                    self.binance.get_account_info()
                    status_text = "Binance: ✅ CONNECTED"
                    status_color = 'green'
                elif hasattr(self.binance, 'client'):
                    # Try basic client test
                    self.binance.client.ping()
                    status_text = "Binance: ✅ CLIENT OK"
                    status_color = 'green'
                else:
                    status_text = "Binance: ⚠️ LOADED"
                    status_color = 'orange'
            except Exception as e:
                status_text = f"Binance: ❌ ERROR"
                status_color = 'red'
                print(f"Binance connection test failed: {e}")
        else:
            status_text = "Binance: 📊 DEMO MODE"
            status_color = 'blue'
        
        self.binance_status_label.config(text=status_text, foreground=status_color)
    
    def test_binance_order(self):
        """✅ ENHANCED: Test completo per ordine Binance"""
        if not self.binance:
            messagebox.showwarning("Test Failed", "Binance integration not available!")
            return
        
        print("\n" + "🧪 BINANCE ORDER TEST" + "="*40)
        
        # Get current price for realistic test
        try:
            symbol = self.current_symbol.get()
            if self.fetch_orderbook():
                current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
                # Use a price 5% below current for BUY test (unlikely to execute immediately)
                test_price = current_price * 0.95
            else:
                test_price = 1.0  # Fallback very low price
            
            print(f"📊 Current price: ${current_price:.2f}")
            print(f"🎯 Test price: ${test_price:.2f} (5% below current)")
            
        except:
            test_price = 1.0
            print("⚠️ Could not get current price, using fallback")
        
        # Create a realistic test order
        test_order = LimitOrder(
            symbol=symbol,
            side='BUY',
            price=test_price,
            quantity=0.001,  # Small quantity
            order_type='LIMIT',
            level_type='test',
            confidence=1.0,
            created_at=datetime.now()
        )
        
        print(f"🧪 Testing order:")
        print(f"   Symbol: {test_order.symbol}")
        print(f"   Side: {test_order.side}")
        print(f"   Price: ${test_order.price:.2f}")
        print(f"   Quantity: {test_order.quantity}")
        
        # Test the order placement
        result = self.place_limit_order(test_order)
        
        print(f"\n📋 Test Result:")
        print(f"   Success: {result}")
        print(f"   Status: {test_order.status}")
        print(f"   Order ID: {test_order.order_id}")
        print("="*60)
        
        # Show result in GUI
        if result:
            messagebox.showinfo("✅ Test Success", 
                f"Test order placed successfully!\n\n"
                f"Status: {test_order.status}\n"
                f"Order ID: {test_order.order_id}\n"
                f"Symbol: {test_order.symbol}\n"
                f"Price: ${test_order.price:.2f}\n"
                f"Quantity: {test_order.quantity}\n\n"
                f"⚠️ Remember to cancel this test order!")
        else:
            messagebox.showerror("❌ Test Failed", 
                f"Test order failed!\n\n"
                f"Status: {test_order.status}\n"
                f"Symbol: {test_order.symbol}\n"
                f"Price: ${test_order.price:.2f}\n"
                f"Quantity: {test_order.quantity}\n\n"
                f"Check console for detailed error information.")
        
        # Add test order to list for tracking
        if result and test_order.status != 'DEMO_PLACED':
            test_order.level_type = 'TEST_ORDER'
            self.pending_orders.append(test_order)
            self.update_orders_display()
            self.update_order_statistics()
    
    def cancel_all_orders(self):
        """Cancella tutti gli ordini - ENHANCED"""
        cancelled_count = 0
        
        for order in self.pending_orders:
            if order.status == 'PLACED' and order.order_id and self.binance:
                try:
                    # Try to cancel real orders
                    if hasattr(self.binance, 'cancel_order'):
                        result = self.binance.cancel_order(
                            symbol=order.symbol,
                            orderId=order.order_id
                        )
                        order.status = 'CANCELLED'
                        cancelled_count += 1
                        print(f"✅ Cancelled order {order.order_id}")
                    elif hasattr(self.binance, 'client') and hasattr(self.binance.client, 'cancel_order'):
                        result = self.binance.client.cancel_order(
                            symbol=order.symbol,
                            orderId=order.order_id
                        )
                        order.status = 'CANCELLED'
                        cancelled_count += 1
                        print(f"✅ Cancelled order {order.order_id}")
                    else:
                        order.status = 'CANCELLED'  # Local cancellation only
                        print(f"⚠️ Local cancellation only for order {order.order_id}")
                        
                except Exception as e:
                    print(f"❌ Failed to cancel order {order.order_id}: {e}")
                    order.status = 'CANCEL_ERROR'
            elif order.status in ['PLACED', 'DEMO_PLACED']:
                order.status = 'CANCELLED'
                cancelled_count += 1
        
        self.update_orders_display()
        self.update_order_statistics()
        
        messagebox.showinfo("Success", 
            f"Cancellation attempted for all orders!\n"
            f"Successfully cancelled: {cancelled_count}")
        
        # Row 2 with test button
        row2 = ttk.Frame(controls_frame)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row2, text="🚀 Place Selected", 
                  command=self.place_selected_orders).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row2, text="❌ Cancel All", 
                  command=self.cancel_all_orders).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row2, text="🧪 Test Binance", 
                  command=self.test_binance_order).pack(side=tk.LEFT, padx=(0, 10))
    
    def clear_orders(self):
        """Pulisce tutti gli ordini pendenti"""
        self.pending_orders = []
        self.update_orders_display()
        self.update_order_statistics()
        self.update_debug_info("All orders cleared")
        messagebox.showinfo("Success", "Tutti gli ordini sono stati rimossi!")
    
    def update_debug_info(self, message):
        """Aggiorna info debug"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self.debug_label.config(text=f"[{timestamp}] {message}"))
    
    def update_config(self):
        """Aggiorna configurazione - ENHANCED"""
        self.config.update({
            'min_volume_threshold': self.min_volume_var.get(),
            'max_distance_pct': self.max_distance_var.get(),
            'wall_multiplier': self.wall_multiplier_var.get(),
            'base_quantity': self.base_quantity_var.get(),
            'min_confidence_limits': self.min_confidence_var.get(),
            'update_interval': self.update_interval_var.get(),
            'min_gap_pct': self.min_gap_var.get(),
            'auto_generate_orders': self.auto_generate_var.get(),
            'auto_place_orders': self.auto_place_var.get(),
            'max_pending_orders': self.max_pending_var.get(),
            'price_tolerance': self.price_tolerance_var.get() / 100,  # Convert to decimal
            'auto_remove_touched': self.auto_remove_touched_var.get(),
            'auto_limit_pending': self.auto_limit_pending_var.get()
        })
        
        # Update instance variables
        self.max_pending_orders = self.config['max_pending_orders']
        self.price_tolerance = self.config['price_tolerance']
        
        print("🔄 Enhanced Configuration updated:")
        for key, value in self.config.items():
            if key in ['min_volume_threshold', 'max_distance_pct', 'wall_multiplier', 
                      'base_quantity', 'min_confidence_limits', 'update_interval', 
                      'min_gap_pct', 'auto_generate_orders', 'auto_place_orders',
                      'max_pending_orders', 'price_tolerance', 'auto_remove_touched', 'auto_limit_pending']:
                print(f"   {key}: {value}")
        
        # Update debug info
        auto_gen = "✅" if self.config['auto_generate_orders'] else "❌"
        auto_place = "✅" if self.config['auto_place_orders'] else "❌"
        auto_remove = "✅" if self.config['auto_remove_touched'] else "❌"
        self.update_debug_info(f"AutoGen: {auto_gen} | AutoPlace: {auto_place} | AutoRemove: {auto_remove} | MaxOrders: {self.max_pending_orders}")
        
        messagebox.showinfo("Success", "Configurazione Enhanced aggiornata!")
    
    def update_charts(self):
        """Aggiorna grafici - ENHANCED"""
        if not self.is_running or not self.orderbook['bids']:
            return
        
        try:
            # Order Book Chart
            self.ax_book.clear()
            
            current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
            
            # Plot order book
            bids = self.orderbook['bids'][:30]
            asks = self.orderbook['asks'][:30]
            
            bid_prices = [p for p, _ in bids]
            bid_volumes = [v for _, v in bids]
            ask_prices = [p for p, _ in asks]
            ask_volumes = [v for _, v in asks]
            
            self.ax_book.barh(bid_prices, bid_volumes, color='green', alpha=0.6, label='Bids')
            self.ax_book.barh(ask_prices, [-v for v in ask_volumes], color='red', alpha=0.6, label='Asks')
            
            # Plot levels
            for level in self.current_levels[:10]:
                color = 'yellow' if level.level_type == 'wall' else 'cyan' if level.level_type == 'gap' else 'white'
                alpha = level.confidence
                
                self.ax_book.axhline(y=level.price, color=color, alpha=alpha, linewidth=2,
                                   linestyle='--' if level.level_type == 'gap' else '-')
            
            # ✅ NEW: Plot pending orders on chart
            pending_orders = [o for o in self.pending_orders if o.status == 'PENDING']
            for order in pending_orders:
                color = 'lime' if order.side == 'BUY' else 'orange'
                self.ax_book.axhline(y=order.price, color=color, alpha=0.8, linewidth=3,
                                   linestyle=':', label=f'{order.side} Orders' if order == pending_orders[0] else "")
            
            self.ax_book.axhline(y=current_price, color='white', linewidth=3, label='Current Price')
            self.ax_book.set_title(f'Order Book + Levels + Orders - {self.current_symbol.get()}', color='white')
            self.ax_book.legend()
            
            # Levels Analysis Chart ENHANCED
            self.ax_levels.clear()
            
            if self.current_levels:
                level_types = [l.level_type for l in self.current_levels]
                type_counts = {}
                for lt in level_types:
                    type_counts[lt] = type_counts.get(lt, 0) + 1
                
                if type_counts:
                    colors = {'support': 'green', 'resistance': 'red', 'wall': 'yellow', 'gap': 'cyan'}
                    bars = self.ax_levels.bar(type_counts.keys(), type_counts.values(),
                                            color=[colors.get(k, 'white') for k in type_counts.keys()],
                                            alpha=0.7)
                    
                    # Aggiungi valori sulle barre
                    for bar, count in zip(bars, type_counts.values()):
                        self.ax_levels.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                                          str(count), ha='center', va='bottom', color='white')
                    
                    self.ax_levels.set_title(f'Levels Distribution (Pending Orders: {len(pending_orders)}/{self.max_pending_orders})', color='white')
                    self.ax_levels.set_ylabel('Count', color='white')
            
            # Styling
            for ax in [self.ax_book, self.ax_levels]:
                ax.set_facecolor('#2d2d2d')
                ax.tick_params(colors='white', labelsize=8)
                ax.grid(True, alpha=0.3)
            
            self.fig_levels.tight_layout(pad=2.0)
            self.canvas_levels.draw()
            
        except Exception as e:
            print(f"Error updating charts: {e}")
    
    def update_levels_display(self):
        """Aggiorna display livelli"""
        # Clear existing items
        for item in self.levels_tree.get_children():
            self.levels_tree.delete(item)
        
        # Add current levels
        for level in self.current_levels:
            values = (
                f"${level.price:.2f}",
                f"{level.volume:.2f}",
                level.side.upper(),
                level.level_type.upper(),
                f"{level.strength:.1f}",
                f"{level.distance_pct:.2f}%",
                f"{level.confidence:.2f}"
            )
            
            item = self.levels_tree.insert('', 'end', values=values)
            
            # Color coding
            if level.level_type == 'wall':
                self.levels_tree.item(item, tags=('wall',))
            elif level.level_type == 'gap':
                self.levels_tree.item(item, tags=('gap',))
            elif level.side == 'bid':
                self.levels_tree.item(item, tags=('support',))
            else:
                self.levels_tree.item(item, tags=('resistance',))
        
        # Configure tags
        self.levels_tree.tag_configure('wall', background='#4a4a00')
        self.levels_tree.tag_configure('gap', background='#004a4a')
        self.levels_tree.tag_configure('support', background='#004a00')
        self.levels_tree.tag_configure('resistance', background='#4a0000')
        
        # Update count
        self.root.after(0, lambda: self.levels_count_label.config(
            text=f"Livelli: {len(self.current_levels)}"
        ))
    
    def update_orders_display(self):
        """Aggiorna display ordini - ENHANCED"""
        # Clear existing items
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)
        
        # Add pending orders
        for order in self.pending_orders:
            # Format timestamps
            created_str = order.created_at.strftime("%H:%M:%S") if order.created_at else "-"
            touched_str = order.touched_at.strftime("%H:%M:%S") if order.touched_at else "-"
            
            values = (
                order.symbol,
                order.side,
                f"${order.price:.2f}",
                f"{order.quantity:.4f}",
                order.order_type,
                order.level_type.upper(),
                f"{order.confidence:.2f}",
                order.status,
                created_str,
                touched_str
            )
            
            item = self.orders_tree.insert('', 'end', values=values)
            
            # Enhanced color coding
            if order.status == 'TOUCHED':
                self.orders_tree.item(item, tags=('touched',))
            elif order.status == 'PLACED':
                self.orders_tree.item(item, tags=('placed',))
            elif order.status == 'ERROR':
                self.orders_tree.item(item, tags=('error',))
            elif order.status == 'CANCELLED':
                self.orders_tree.item(item, tags=('cancelled',))
            elif order.side == 'BUY':
                self.orders_tree.item(item, tags=('buy',))
            else:
                self.orders_tree.item(item, tags=('sell',))
        
        # Configure enhanced tags
        self.orders_tree.tag_configure('touched', background='#4a4a00', foreground='white')  # Yellow
        self.orders_tree.tag_configure('placed', background='#004a00', foreground='white')   # Green
        self.orders_tree.tag_configure('error', background='#4a0000', foreground='white')    # Red
        self.orders_tree.tag_configure('cancelled', background='#2a2a2a', foreground='gray') # Gray
        self.orders_tree.tag_configure('buy', background='#002a00', foreground='lightgreen') # Dark green
        self.orders_tree.tag_configure('sell', background='#2a0000', foreground='lightcoral') # Dark red
        
        # Update statistics
        self.update_order_statistics()
    
    def set_aggressive_preset(self):
        """Preset aggressivo - molti ordini"""
        self.min_volume_var.set(0.5)
        self.max_distance_var.set(10.0)
        self.wall_multiplier_var.set(1.0)
        self.min_confidence_var.set(0.05)
        self.min_gap_var.set(0.02)
        self.max_pending_var.set(10)  # ✅ NEW
        self.price_tolerance_var.set(0.5)  # ✅ NEW
        self.update_config()
        messagebox.showinfo("Preset", "🔥 Configurazione AGGRESSIVA applicata!")
    
    def set_balanced_preset(self):
        """Preset bilanciato - configurazione media"""
        self.min_volume_var.set(2.0)
        self.max_distance_var.set(3.0)
        self.wall_multiplier_var.set(2.0)
        self.min_confidence_var.set(0.30)
        self.min_gap_var.set(0.1)
        self.max_pending_var.set(5)  # ✅ NEW
        self.price_tolerance_var.set(0.2)  # ✅ NEW
        self.update_config()
        messagebox.showinfo("Preset", "⚖️ Configurazione BILANCIATA applicata!")
    
    def set_conservative_preset(self):
        """Preset conservativo - pochi ordini di qualità"""
        self.min_volume_var.set(10.0)
        self.max_distance_var.set(1.0)
        self.wall_multiplier_var.set(5.0)
        self.min_confidence_var.set(0.70)
        self.min_gap_var.set(0.2)
        self.max_pending_var.set(3)  # ✅ NEW
        self.price_tolerance_var.set(0.1)  # ✅ NEW
        self.update_config()
        messagebox.showinfo("Preset", "🛡️ Configurazione CONSERVATIVA applicata!")
    
    def set_no_limits_preset(self):
        """Preset senza limiti - genera tutto"""
        self.min_volume_var.set(0.1)
        self.max_distance_var.set(20.0)
        self.wall_multiplier_var.set(0.5)
        self.min_confidence_var.set(0.01)
        self.min_gap_var.set(0.01)
        self.max_pending_var.set(20)  # ✅ NEW
        self.price_tolerance_var.set(1.0)  # ✅ NEW
        self.update_config()
        messagebox.showinfo("Preset", "🚀 Configurazione NO LIMITS applicata!")
    
    def analysis_loop(self):
        """Loop principale analisi - ENHANCED"""
        while self.is_running:
            try:
                # Fetch order book
                if self.fetch_orderbook():
                    current_price = (self.orderbook['bids'][0][0] + self.orderbook['asks'][0][0]) / 2
                    
                    # ✅ ENHANCED: Check for touched orders FIRST
                    if self.pending_orders:
                        touched_orders = OrderManager.check_orders_touched(
                            self.pending_orders, current_price, self.price_tolerance
                        )
                        
                        if touched_orders:
                            print(f"🎯 Found {len(touched_orders)} touched orders at price ${current_price:.2f}")
                            
                            # Auto remove if enabled
                            if self.config.get('auto_remove_touched', True):
                                removed_count = OrderManager.remove_touched_orders(self.pending_orders)
                                if removed_count > 0:
                                    print(f"🗑️ Auto-removed {removed_count} touched orders")
                                    self.root.after(0, self.update_orders_display)
                    
                    # ✅ ENHANCED: Limit pending orders if enabled
                    if self.config.get('auto_limit_pending', True):
                        current_pending = len([o for o in self.pending_orders if o.status == 'PENDING'])
                        if current_pending > self.max_pending_orders:
                            removed_count = OrderManager.limit_pending_orders(self.pending_orders, self.max_pending_orders)
                            if removed_count > 0:
                                print(f"📊 Auto-limited to {self.max_pending_orders} orders, removed {removed_count}")
                                self.root.after(0, self.update_orders_display)
                    
                    # Analyze levels
                    levels = self.analyze_levels()
                    self.current_levels = levels
                    
                    # Update displays
                    self.update_levels_display()
                    self.update_charts()
                    
                    # ✅ ENHANCED: Auto generate orders with smart limiting
                    if levels and self.config.get('auto_generate_orders', True):
                        current_pending = len([o for o in self.pending_orders if o.status == 'PENDING'])
                        available_slots = max(0, self.max_pending_orders - current_pending)
                        
                        if available_slots > 0:
                            # Generate only what we have space for
                            new_orders = LimitOrderStrategy.generate_limit_orders(
                                levels, current_price, self.config, max_orders=available_slots
                            )
                            
                            if new_orders:
                                print(f"🎯 Auto-generated {len(new_orders)} orders from {len(levels)} levels (slots: {available_slots})")
                                
                                # Remove duplicates (same price)
                                existing_prices = set(order.price for order in self.pending_orders)
                                unique_orders = [order for order in new_orders if order.price not in existing_prices]
                                
                                if unique_orders:
                                    self.pending_orders.extend(unique_orders)
                                    print(f"📋 Added {len(unique_orders)} unique orders to pending list")
                                    
                                    # Update display
                                    self.root.after(0, self.update_orders_display)
                                    
                                    # Auto place if enabled
                                    if self.config.get('auto_place_orders', False):
                                        high_conf_orders = [o for o in unique_orders if o.confidence >= 0.8]
                                        placed_count = 0
                                        
                                        for order in high_conf_orders:
                                            placed_orders_count = len([o for o in self.pending_orders if o.status == 'PLACED'])
                                            if placed_orders_count < 10:  # Max 10 placed orders
                                                if self.place_limit_order(order):
                                                    placed_count += 1
                                        
                                        if placed_count > 0:
                                            print(f"🚀 Auto-placed {placed_count} high confidence orders")
                                            self.root.after(0, self.update_orders_display)
                                else:
                                    print("⚠️ No new unique orders to add (duplicates filtered)")
                        else:
                            print(f"📊 No available slots for new orders ({current_pending}/{self.max_pending_orders})")
                    else:
                        if not levels:
                            print("📊 No levels detected in current analysis")
                    
                    # Update statistics
                    self.root.after(0, self.update_order_statistics)
                    
                    # Update status
                    stats = OrderManager.get_order_statistics(self.pending_orders)
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"Sistema: ATTIVO ✅ (P:{stats['pending']}/T:{stats['touched']})", 
                        foreground='green'
                    ))
                    
                else:
                    self.root.after(0, lambda: self.status_label.config(
                        text="Sistema: ERRORE API ❌", foreground='red'
                    ))
                
            except Exception as e:
                print(f"Error in enhanced analysis loop: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_label.config(
                    text="Sistema: ERRORE ❌", foreground='red'
                ))
            
            # Wait for next update
            time.sleep(self.config.get('update_interval', 10))
    
    def start_analysis(self):
        """Avvia analisi - ENHANCED"""
        if not self.current_symbol.get():
            messagebox.showerror("Error", "Seleziona un simbolo!")
            return
        
        # Test connection
        if not self.fetch_orderbook():
            messagebox.showerror("Error", "Impossibile connettersi a Binance API!")
            return
        
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Clear data
        self.current_levels = []
        self.price_history.clear()
        
        # Update config
        self.config['symbol'] = self.current_symbol.get()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        print(f"🚀 Enhanced Order Book Levels Analysis started for {self.current_symbol.get()}")
        print(f"   Min Volume: {self.config['min_volume_threshold']}")
        print(f"   Max Distance: {self.config['max_distance_pct']}%")
        print(f"   Max Pending Orders: {self.max_pending_orders}")
        print(f"   Price Tolerance: {self.price_tolerance:.3f}")
        print(f"   Auto Remove Touched: {'✅' if self.config.get('auto_remove_touched', True) else '❌'}")
        print(f"   Auto Limit Pending: {'✅' if self.config.get('auto_limit_pending', True) else '❌'}")
        print(f"   Auto Orders: {'✅' if self.config['auto_place_orders'] else '❌'}")
    
    def stop_analysis(self):
        """Ferma analisi"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self.status_label.config(text="Sistema: FERMO ⏹️", foreground='red')
        print("⏹️ Enhanced Order Book Levels Analysis stopped")
        
        # Show final statistics
        stats = OrderManager.get_order_statistics(self.pending_orders)
        print(f"📊 Final Statistics:")
        print(f"   Total Orders: {stats['total']}")
        print(f"   Pending: {stats['pending']}")
        print(f"   Touched: {stats['touched']}")
        print(f"   Placed: {stats['placed']}")
        
        # Cancel pending orders if requested
        if self.pending_orders:
            reply = messagebox.askyesno("Cancel Orders", 
                                      f"Cancellare {len(self.pending_orders)} ordini totali?\n"
                                      f"Pending: {stats['pending']}, Touched: {stats['touched']}")
            if reply:
                self.cancel_all_orders()
    
    def run(self):
        """Avvia applicazione - ENHANCED"""
        print("🚀 Enhanced Order Book Levels Analyzer")
        print("=" * 60)
        print("📊 Funzionalità ENHANCED:")
        print("   ✅ Analisi livelli significativi nel book")
        print("   ✅ Identificazione supporti/resistenze")
        print("   ✅ Rilevamento muri e gap")
        print("   ✅ Generazione ordini limite automatici")
        print("   ✅ Integrazione Binance per trading live")
        print("   🆕 LIMITE MASSIMO ORDINI PENDENTI (configurabile)")
        print("   🆕 RIMOZIONE AUTOMATICA ORDINI TOCCATI")
        print("   🆕 MONITORAGGIO PREZZO CON TOLLERANZA")
        print("   🆕 STATISTICHE ORDINI IN TEMPO REALE")
        print()
        print("🎯 Strategia Enhanced:")
        print("   📈 BUY sui supporti forti")
        print("   📉 SELL sulle resistenze forti")
        print("   🕳️ Ordini sui gap per catturare movimenti")
        print("   🧱 Priorità ai muri di liquidità")
        print("   🎯 MAX 5 ordini pendenti (default)")
        print("   🗑️ Auto-rimozione ordini toccati dal prezzo")
        print()
        print("⚙️ Configurabile Enhanced:")
        print("   🔧 Soglie volume e distanza")
        print("   🎯 Confidence minima per ordini")
        print("   💰 Quantità base personalizzabile")
        print("   🤖 Piazzamento automatico ordini")
        print("   📊 Limite ordini pendenti (1-50)")
        print("   🎯 Tolleranza prezzo per rilevamento touch")
        print("   🗑️ Auto-rimozione ordini toccati")
        print("=" * 60)
        
        self.root.mainloop()

# === MAIN ===
if __name__ == "__main__":
    try:
        app = OrderBookLevelsGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Assicurati che advanced_binance_integration.py sia disponibile")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    input("\nPremi INVIO per uscire...")