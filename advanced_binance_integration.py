#!/usr/bin/env python3
"""
Integrazione Binance Avanzata per Trader Esperti
Accesso diretto al conto live con funzionalità complete
"""

import hmac
import hashlib
import time
import requests
from urllib.parse import urlencode
import json
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
import threading
from datetime import datetime
import logging

# Setup logging per debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBinanceAPI:
    """API Binance avanzata per trader esperti"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': api_key})
        
        # Cache per symbol info
        self._symbol_info_cache = {}
        self._account_info_cache = None
        self._cache_timestamp = 0
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        
        logger.info(f"Binance API inizializzata - {'TESTNET' if testnet else 'LIVE'}")
    
    def _generate_signature(self, params: dict) -> str:
        """Genera firma HMAC-SHA256"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """Richiesta HTTP con rate limiting e error handling"""
        if params is None:
            params = {}
        
        # Rate limiting semplice
        current_time = time.time()
        if current_time - self._last_request_time < 0.1:  # 100ms min between requests
            time.sleep(0.1)
        
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            response = self.session.request(method, url, params=params, timeout=10)
            self._last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.text else {}
                raise Exception(f"API Error {response.status_code}: {error_data.get('msg', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {e}")
    
    # === ACCOUNT & BALANCE ===
    
    def get_account_info(self, force_refresh: bool = False) -> dict:
        """Info account con caching intelligente"""
        current_time = time.time()
        
        if not force_refresh and self._account_info_cache and (current_time - self._cache_timestamp < 30):
            return self._account_info_cache
        
        account_info = self._request('GET', '/api/v3/account', signed=True)
        self._account_info_cache = account_info
        self._cache_timestamp = current_time
        
        return account_info
    
    def get_balance(self, asset: str) -> Dict[str, float]:
        """Balance specifico asset"""
        account = self.get_account_info()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return {
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
        return {'free': 0.0, 'locked': 0.0, 'total': 0.0}
    
    def get_all_balances(self, min_value: float = 0.0) -> Dict[str, Dict[str, float]]:
        """Tutti i balance con valore > min_value"""
        account = self.get_account_info()
        balances = {}
        
        for balance in account['balances']:
            total = float(balance['free']) + float(balance['locked'])
            if total > min_value:
                balances[balance['asset']] = {
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': total
                }
        
        return balances
    
    # === SYMBOL INFO & PRICE ===
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Info simbolo con caching"""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        
        exchange_info = self._request('GET', '/api/v3/exchangeInfo')
        
        for s in exchange_info['symbols']:
            self._symbol_info_cache[s['symbol']] = s
        
        return self._symbol_info_cache.get(symbol, {})
    
    def get_ticker_price(self, symbol: str) -> float:
        """Prezzo corrente simbolo"""
        ticker = self._request('GET', '/api/v3/ticker/price', {'symbol': symbol})
        return float(ticker['price'])
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> dict:
        """Order book"""
        return self._request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': limit})
    
    # === FORMATO QUANTITÀ ===
    
    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatta quantità secondo regole simbolo"""
        symbol_info = self.get_symbol_info(symbol)
        
        if not symbol_info:
            return f"{quantity:.8f}"
        
        # Trova LOT_SIZE filter
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                
                # Calcola precision necessaria
                precision = len(f['stepSize'].rstrip('0').split('.')[-1]) if '.' in f['stepSize'] else 0
                
                # Arrotonda down alla step size
                formatted_qty = Decimal(str(quantity)) - (Decimal(str(quantity)) % Decimal(str(step_size)))
                
                return f"{formatted_qty:.{precision}f}"
        
        return f"{quantity:.8f}"
    
    def format_price(self, symbol: str, price: float) -> str:
        """Formatta prezzo secondo regole simbolo"""
        symbol_info = self.get_symbol_info(symbol)
        
        if not symbol_info:
            return f"{price:.8f}"
        
        # Trova PRICE_FILTER
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick_size = float(f['tickSize'])
                precision = len(f['tickSize'].rstrip('0').split('.')[-1]) if '.' in f['tickSize'] else 0
                
                # Arrotonda al tick size
                formatted_price = Decimal(str(price)) - (Decimal(str(price)) % Decimal(str(tick_size)))
                
                return f"{formatted_price:.{precision}f}"
        
        return f"{price:.8f}"
    
    def calculate_max_quantity(self, symbol: str, balance_asset: str, price: float = None) -> float:
        """Calcola quantità massima acquistabile"""
        balance = self.get_balance(balance_asset)
        available = balance['free']
        
        if price is None:
            price = self.get_ticker_price(symbol)
        
        # Calcola quantità massima
        max_qty = available / price * 0.999  # 0.1% buffer per commissioni
        
        # Applica filtri simbolo
        symbol_info = self.get_symbol_info(symbol)
        
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                max_qty_limit = float(f['maxQty'])
                
                max_qty = min(max_qty, max_qty_limit)
                
                if max_qty < min_qty:
                    return 0.0
        
        return max_qty
    
    # === TRADING ORDERS ===
    
    def place_market_order(self, symbol: str, side: str, quantity: float, 
                          quote_order_qty: float = None) -> dict:
        """Ordine market con gestione quantità intelligente"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET'
        }
        
        if quote_order_qty:
            # Ordine per valore in quote asset (es. USDT)
            params['quoteOrderQty'] = self.format_price(symbol, quote_order_qty)
        else:
            # Ordine per quantità base asset
            params['quantity'] = self.format_quantity(symbol, quantity)
        
        return self._request('POST', '/api/v3/order', params, signed=True)
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float,
                         time_in_force: str = 'GTC') -> dict:
        """Ordine limit"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': price
        }
        
        return self._request('POST', '/api/v3/order', params, signed=True)
    
    def place_stop_loss_order(self, symbol: str, side: str, quantity: float, 
                             stop_price: float, limit_price: float = None) -> dict:
        """Ordine stop loss / stop limit"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'quantity': self.format_quantity(symbol, quantity),
            'stopPrice': self.format_price(symbol, stop_price)
        }
        
        if limit_price:
            params['type'] = 'STOP_LOSS_LIMIT'
            params['timeInForce'] = 'GTC'
            params['price'] = self.format_price(symbol, limit_price)
        else:
            params['type'] = 'STOP_LOSS'
        
        return self._request('POST', '/api/v3/order', params, signed=True)
    
    def place_oco_order(self, symbol: str, side: str, quantity: float,
                       price: float, stop_price: float, stop_limit_price: float) -> dict:
        """Ordine OCO (One-Cancels-Other)"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'quantity': self.format_quantity(symbol, quantity),
            'price': self.format_price(symbol, price),
            'stopPrice': self.format_price(symbol, stop_price),
            'stopLimitPrice': self.format_price(symbol, stop_limit_price),
            'stopLimitTimeInForce': 'GTC'
        }
        
        return self._request('POST', '/api/v3/order/oco', params, signed=True)
    
    # === ORDER MANAGEMENT ===
    
    def get_open_orders(self, symbol: str = None) -> List[dict]:
        """Ordini aperti"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/api/v3/openOrders', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> dict:
        """Cancella ordine"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Specificare order_id o orig_client_order_id")
        
        return self._request('DELETE', '/api/v3/order', params, signed=True)
    
    def cancel_all_orders(self, symbol: str) -> List[dict]:
        """Cancella tutti gli ordini per un simbolo"""
        params = {'symbol': symbol}
        return self._request('DELETE', '/api/v3/openOrders', params, signed=True)
    
    def get_order_status(self, symbol: str, order_id: int) -> dict:
        """Status ordine"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._request('GET', '/api/v3/order', params, signed=True)
    
    # === TRADING HISTORY ===
    
    def get_trade_history(self, symbol: str, limit: int = 500) -> List[dict]:
        """Storico trades"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        return self._request('GET', '/api/v3/myTrades', params, signed=True)
    
    def get_order_history(self, symbol: str, limit: int = 500) -> List[dict]:
        """Storico ordini"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        return self._request('GET', '/api/v3/allOrders', params, signed=True)

class SmartPositionManager:
    """Gestione posizioni intelligente con controlli migliorati"""
    
    def __init__(self, binance_api):
        self.api = binance_api
        self.positions = {}
        
    def get_tradeable_balance(self, symbol: str, side: str) -> dict:
        """Ottiene balance utilizzabile per il trade"""
        
        # Estrae base e quote asset dal simbolo
        # Es: BTCUSDC -> base=BTC, quote=USDC
        symbol_info = self.api.get_symbol_info(symbol)
        
        if not symbol_info:
            return {'error': f'Symbol {symbol} not found'}
        
        base_asset = symbol_info['baseAsset']    # BTC
        quote_asset = symbol_info['quoteAsset']  # USDC
        
        if side.upper() == 'BUY':
            # Per comprare, serve quote asset (USDC)
            balance = self.api.get_balance(quote_asset)
            return {
                'asset': quote_asset,
                'available': balance['free'],
                'type': 'quote'
            }
        else:  # SELL
            # Per vendere, serve base asset (BTC)
            balance = self.api.get_balance(base_asset)
            return {
                'asset': base_asset,
                'available': balance['free'],
                'type': 'base'
            }
    
    def calculate_smart_quantity(self, symbol: str, side: str, usd_amount: float, 
                               current_price: float) -> dict:
        """Calcola quantità intelligente basata su balance disponibile"""
        
        try:
            balance_info = self.get_tradeable_balance(symbol, side)
            
            if 'error' in balance_info:
                return balance_info
            
            available = balance_info['available']
            asset = balance_info['asset']
            
            if side.upper() == 'BUY':
                # Comprare: usa USDC disponibili
                max_usd_available = available * 0.999  # 0.1% buffer per fees
                
                # Usa il minore tra richiesto e disponibile
                trade_usd_amount = min(usd_amount, max_usd_available)
                
                if trade_usd_amount < 10:  # Minimo $10 per trade
                    return {'error': f'Insufficient {asset} balance. Available: ${available:.2f}, Required: ${usd_amount:.2f}'}
                
                # Calcola quantità in base asset
                quantity = trade_usd_amount / current_price
                
            else:  # SELL
                # Vendere: usa crypto disponibili  
                if available <= 0:
                    return {'error': f'No {asset} balance to sell. Available: {available:.8f}'}
                
                # Calcola valore in USD del balance disponibile
                available_usd_value = available * current_price
                
                # Usa il minore tra richiesto e disponibile
                trade_usd_amount = min(usd_amount, available_usd_value * 0.999)
                
                if trade_usd_amount < 10:  # Minimo $10 per trade
                    return {'error': f'Insufficient {asset} balance. Available value: ${available_usd_value:.2f}, Required: ${usd_amount:.2f}'}
                
                # Calcola quantità da vendere
                quantity = trade_usd_amount / current_price
            
            # Formatta quantità secondo regole simbolo
            formatted_quantity = float(self.api.format_quantity(symbol, quantity))
            
            # Verifica quantità minima
            symbol_info = self.api.get_symbol_info(symbol)
            for f in symbol_info.get('filters', []):
                if f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    if formatted_quantity < min_qty:
                        return {'error': f'Quantity {formatted_quantity:.8f} below minimum {min_qty:.8f}'}
            
            return {
                'quantity': formatted_quantity,
                'usd_amount': trade_usd_amount,
                'asset_used': asset,
                'available_balance': available
            }
            
        except Exception as e:
            return {'error': f'Error calculating quantity: {e}'}
    
    def execute_smart_entry(self, symbol: str, side: str, usd_amount: float,
                           stop_loss_pct: float, take_profit_pct: float,
                           order_type: str = 'MARKET') -> dict:
        """Esecuzione entry intelligente con controlli migliorati"""
        
        try:
            # Ottieni prezzo corrente
            current_price = self.api.get_ticker_price(symbol)
            
            # Calcola quantità intelligente
            qty_result = self.calculate_smart_quantity(symbol, side, usd_amount, current_price)
            
            if 'error' in qty_result:
                return qty_result
            
            quantity = qty_result['quantity']
            actual_usd = qty_result['usd_amount']
            
            # Calcola stop loss e take profit
            if side.upper() == 'BUY':
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                take_profit_price = current_price * (1 + take_profit_pct / 100)
            else:
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)
                take_profit_price = current_price * (1 - take_profit_pct / 100)
            
            # Log dettagli pre-trade
            print(f"📊 TRADE DETAILS:")
            print(f"   Balance available: {qty_result['available_balance']:.8f} {qty_result['asset_used']}")
            print(f"   Requested USD: ${usd_amount:.2f}")
            print(f"   Actual USD: ${actual_usd:.2f}")
            print(f"   Quantity: {quantity:.8f}")
            print(f"   Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct}%)")
            print(f"   Take Profit: ${take_profit_price:.2f} ({take_profit_pct}%)")
            
            # Esegui ordine entry
            if order_type.upper() == 'MARKET':
                entry_result = self.api.place_market_order(symbol, side, quantity)
            else:
                entry_result = self.api.place_limit_order(symbol, side, quantity, current_price)
            
            if 'orderId' in entry_result:
                # Salva posizione
                position_info = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'entry_order_id': entry_result['orderId'],
                    'usd_amount': actual_usd,
                    'timestamp': datetime.now()
                }
                
                self.positions[symbol] = position_info
                
                # Piazza ordini stop loss e take profit (opzionale)
                # self._place_exit_orders(position_info)
                
                return {
                    'success': True,
                    'entry_order': entry_result,
                    'position_info': position_info,
                    'quantity_details': qty_result
                }
            
            return {'error': 'Failed to place entry order', 'response': entry_result}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_position_status(self, symbol: str) -> dict:
        """Status posizione corrente"""
        if symbol not in self.positions:
            return {'error': 'Position not found'}
        
        position = self.positions[symbol]
        
        try:
            # Ottieni prezzo corrente
            current_price = self.api.get_ticker_price(symbol)
            
            # Calcola P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            if position['side'] == 'BUY':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            # Calcola percentuali
            pnl_pct = (unrealized_pnl / (entry_price * quantity)) * 100
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'pnl_percentage': pnl_pct,
                'position_info': position
            }
            
        except Exception as e:
            return {'error': str(e)}

class BinanceTradingIntegration:
    """Integrazione completa con configurazione quantità migliorata"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api = AdvancedBinanceAPI(api_key, api_secret, testnet)
        self.position_manager = SmartPositionManager(self.api)
        
        # 🔧 CONFIGURAZIONE MIGLIORATA CON CONTROLLI QUANTITÀ
        self.config = {
            'auto_trade_enabled': False,
            
            # === GESTIONE QUANTITÀ ===
            'trade_mode': 'fixed_usd',           # 'fixed_usd' o 'percentage_balance'
            'fixed_usd_amount': 30.0,            # USD fissi per trade
            'percentage_balance': 5.0,           # % del balance per trade
            'min_trade_usd': 10.0,               # Minimo USD per trade
            'max_trade_usd': 100.0,              # Massimo USD per trade
            
            # === RISK MANAGEMENT ===
            'default_stop_loss_pct': 2.5,
            'default_take_profit_pct': 5.0,
            'max_open_positions': 3,
            'min_confidence_threshold': 0.80,
            
            # === ORDINI ===
            'order_type': 'MARKET',              # MARKET o LIMIT
            'enable_stop_loss_orders': False,    # OCO orders automatici
            
            # === ASSET FILTERING ===
            'allowed_quote_assets': ['USDT', 'USDC', 'BUSD'],  # Quote assets permessi
            'blocked_symbols': [],               # Simboli bloccati
        }
        
        # Tracking
        self.trade_log = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
    
    def update_config(self, new_config: dict):
        """Aggiorna configurazione"""
        self.config.update(new_config)
        logger.info(f"Configuration updated: {new_config}")
    
    def calculate_trade_amount(self, symbol: str, side: str) -> float:
        """Calcola amount per il trade basato su configurazione"""
        
        try:
            if self.config['trade_mode'] == 'fixed_usd':
                # Importo fisso in USD
                return self.config['fixed_usd_amount']
            
            elif self.config['trade_mode'] == 'percentage_balance':
                # Percentuale del balance
                balance_info = self.position_manager.get_tradeable_balance(symbol, side)
                
                if 'error' in balance_info:
                    return self.config['fixed_usd_amount']  # Fallback
                
                available = balance_info['available']
                
                if balance_info['type'] == 'quote':
                    # Balance già in USD (USDC/USDT)
                    trade_amount = available * (self.config['percentage_balance'] / 100)
                else:
                    # Balance in crypto, converti in USD
                    current_price = self.api.get_ticker_price(symbol)
                    available_usd = available * current_price
                    trade_amount = available_usd * (self.config['percentage_balance'] / 100)
                
                # Applica limiti
                trade_amount = max(self.config['min_trade_usd'], trade_amount)
                trade_amount = min(self.config['max_trade_usd'], trade_amount)
                
                return trade_amount
            
            return self.config['fixed_usd_amount']  # Default fallback
            
        except Exception as e:
            print(f"Error calculating trade amount: {e}")
            return self.config['fixed_usd_amount']
    
    def is_symbol_tradeable(self, symbol: str) -> bool:
        """Verifica se simbolo è tradeable"""
        
        # Controlla simboli bloccati
        if symbol in self.config['blocked_symbols']:
            return False
        
        # Controlla quote asset permessi
        try:
            symbol_info = self.api.get_symbol_info(symbol)
            quote_asset = symbol_info.get('quoteAsset', '')
            
            if quote_asset not in self.config['allowed_quote_assets']:
                return False
            
            # Controlla se simbolo è attivo
            if symbol_info.get('status') != 'TRADING':
                return False
            
            return True
            
        except:
            return False
    
    def execute_signal(self, signal) -> dict:
        """Esegue segnale con controlli migliorati"""
        
        if not self.config['auto_trade_enabled']:
            return {'status': 'skipped', 'reason': 'Auto trading disabled'}
        
        if signal.confidence < self.config['min_confidence_threshold']:
            return {'status': 'skipped', 'reason': f'Confidence too low: {signal.confidence:.2f}'}
        
        if not self.is_symbol_tradeable(signal.symbol):
            return {'status': 'skipped', 'reason': f'Symbol {signal.symbol} not tradeable'}
        
        # Verifica numero massimo posizioni
        open_positions = len([p for p in self.position_manager.positions.values()])
        if open_positions >= self.config['max_open_positions']:
            return {'status': 'skipped', 'reason': 'Max positions reached'}
        
        try:
            # Calcola trade amount
            trade_amount = self.calculate_trade_amount(signal.symbol, signal.action)
            
            # Aggiusta per confidence
            adjusted_amount = trade_amount * signal.confidence
            
            print(f"💰 TRADE AMOUNT CALCULATION:")
            print(f"   Mode: {self.config['trade_mode']}")
            print(f"   Base amount: ${trade_amount:.2f}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Final amount: ${adjusted_amount:.2f}")
            
            # Esegui trade
            result = self.position_manager.execute_smart_entry(
                symbol=signal.symbol,
                side=signal.action,
                usd_amount=adjusted_amount,
                stop_loss_pct=self.config['default_stop_loss_pct'],
                take_profit_pct=self.config['default_take_profit_pct'],
                order_type=self.config['order_type']
            )
            
            # Log trade
            trade_record = {
                'timestamp': datetime.now(),
                'signal': signal,
                'result': result,
                'trade_amount': adjusted_amount
            }
            
            self.trade_log.append(trade_record)
            self.total_trades += 1
            
            if result.get('success'):
                print(f"✅ Trade executed successfully!")
                return {'status': 'executed', 'result': result}
            else:
                print(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                return {'status': 'failed', 'error': result.get('error')}
                
        except Exception as e:
            print(f"❌ Exception executing signal: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_portfolio_summary(self) -> dict:
        """Summary portfolio completo"""
        try:
            # Balance account
            balances = self.api.get_all_balances(min_value=0.01)
            
            # Posizioni aperte
            open_positions = {}
            total_unrealized_pnl = 0.0
            
            for symbol, position in self.position_manager.positions.items():
                status = self.position_manager.get_position_status(symbol)
                if 'unrealized_pnl' in status:
                    open_positions[symbol] = status
                    total_unrealized_pnl += status['unrealized_pnl']
            
            # Stats trading
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            return {
                'balances': balances,
                'open_positions': open_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_realized_pnl': self.total_pnl
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def emergency_close_all(self) -> dict:
        """Chiude tutte le posizioni - EMERGENZA"""
        results = []
        
        for symbol in list(self.position_manager.positions.keys()):
            try:
                # Cancella tutti gli ordini aperti
                cancel_result = self.api.cancel_all_orders(symbol)
                
                # Chiudi posizione a mercato
                position = self.position_manager.positions[symbol]
                exit_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                
                close_result = self.api.place_market_order(
                    symbol, exit_side, position['quantity']
                )
                
                results.append({
                    'symbol': symbol,
                    'cancel_orders': cancel_result,
                    'close_position': close_result
                })
                
                # Rimuovi dalla tracking
                del self.position_manager.positions[symbol]
                
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return {'emergency_close_results': results}

# === ESEMPIO USO ===
if __name__ == "__main__":
    # 🔑 INSERISCI LE TUE API QUI
    # Inizializzazione
    API_KEY = "seU99BIqWSVbtZ8PmW0PTnNSLpWsj8WE43JFKwzLHPGu7Wb4ZFwE6fjddljcGK87"
    API_SECRET = "0snc4bvVMlK0OlSahiW0grsMrRAzapDj17J99gGxokes1LRZyi2NEs9n4vMJ6iVx"
    
    # Per testnet usa testnet=True
    trading = BinanceTradingIntegration(API_KEY, API_SECRET, testnet=False)
    
    # Configurazione
    trading.update_config({
        'auto_trade_enabled': True,
        'fixed_usd_amount': 25.0,
        'max_open_positions': 3,
        'min_confidence_threshold': 0.85,
        'blocked_symbols': ['BTCUSDC'],  # Blocca simboli senza balance
        'allowed_quote_assets': ['USDC', 'USDT']
    })
    
    # Test connessione
    try:
        portfolio = trading.get_portfolio_summary()
        print("✅ Connessione riuscita!")
        print(f"Portfolio assets: {len(portfolio.get('balances', {}))}")
        print(f"Total trades: {portfolio.get('total_trades', 0)}")
    except Exception as e:
        print(f"❌ Errore connessione: {e}")

"""
🔧 INTEGRAZIONE NEL TUO BOT ESISTENTE:

1. Sostituisci advanced_binance_integration.py con questo codice

2. Nel tuo orderbook_monitor_v0.py, modifica setup_binance_live_trading():

def setup_binance_live_trading(self):
    # 🔑 INSERISCI LE TUE API QUI
    API_KEY = "la_tua_api_key_qui"
    API_SECRET = "il_tuo_secret_qui"
    
    try:
        print("🚀 Inizializzazione Binance LIVE Trading...")
        
        self.binance_trading = BinanceTradingIntegration(
            API_KEY, API_SECRET, testnet=False
        )
        
        # 📊 CONFIGURAZIONE OTTIMIZZATA
        self.binance_trading.update_config({
            'auto_trade_enabled': True,
            
            # === GESTIONE QUANTITÀ ===
            'trade_mode': 'fixed_usd',
            'fixed_usd_amount': 20.0,           # $20 per trade
            'min_trade_usd': 10.0,
            'max_trade_usd': 50.0,
            
            # === RISK MANAGEMENT ===  
            'default_stop_loss_pct': 2.5,
            'default_take_profit_pct': 5.0,
            'max_open_positions': 2,
            'min_confidence_threshold': 0.85,
            
            # === ASSET FILTERING ===
            'allowed_quote_assets': ['USDC', 'USDT'],
            'blocked_symbols': ['BTCUSDC'],     # Blocca BTC se non hai balance
            
            # === ORDINI ===
            'order_type': 'MARKET',
        })
        
        # Test portfolio
        portfolio = self.binance_trading.get_portfolio_summary()
        
        print("✅ BINANCE LIVE TRADING ATTIVATO!")
        print(f"💰 Portfolio assets: {len(portfolio.get('balances', {}))}")
        print(f"💵 Trade amount: ${self.binance_trading.config['fixed_usd_amount']}")
        print(f"📊 Min confidence: {self.binance_trading.config['min_confidence_threshold']*100}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore setup Binance: {e}")
        self.binance_trading = None
        return False

3. Modifica execute_signal() per includere trading live:

def execute_signal(self, signal: TradingSignal):
    # Simulazione esistente
    self.execute_signal_simulation(signal)
    
    # Trading reale Binance
    if hasattr(self, 'binance_trading') and self.binance_trading:
        live_result = self.binance_trading.execute_signal(signal)
        
        if live_result['status'] == 'executed':
            print(f"✅ LIVE TRADE SUCCESS: {signal.symbol} {signal.action}")
        elif live_result['status'] == 'failed':
            print(f"❌ LIVE TRADE FAILED: {live_result.get('error')}")
        elif live_result['status'] == 'skipped':
            print(f"⏸️ LIVE TRADE SKIPPED: {live_result.get('reason')}")

🎯 CONFIGURAZIONE CONSIGLIATA BASATA SUL TUO PORTFOLIO:

# Simboli che puoi tradare (hai i balance):
'allowed_symbols': [
    'ETHUSDC',    # Hai ETH
    'UNIUSDT',    # Hai UNI  
    'MANAUSDT',   # Hai MANA
    'AUDIOUSDT',  # Hai AUDIO
    'XRPUSDT',    # Hai XRP
    'CRVUSDT',    # Hai CRV
    'SUIUSDT',    # Hai SUI
]

# Simboli da evitare (non hai balance):
'blocked_symbols': [
    'BTCUSDC',    # Non hai BTC
    'BNBUSDT',    # Poco BNB
    'ADAUSDT',    # Non hai ADA
    'SOLUSDT',    # Non hai SOL
]

🚀 FEATURES IMPLEMENTATE:
✅ Controllo balance intelligente
✅ Calcolo quantità automatico
✅ Asset filtering avanzato
✅ Risk management per trade
✅ Position tracking
✅ Emergency close all
✅ Portfolio summary
✅ Error handling robusto
✅ Configurazione flessibile
✅ Logging dettagliato

💡 NEXT STEPS:
1. Sostituisci il file advanced_binance_integration.py
2. Aggiungi le tue API credentials
3. Configura blocked_symbols per simboli senza balance
4. Testa con fixed_usd_amount basso ($15-20)
5. Monitora i primi trades attentamente
"""









"""
    # Inizializzazione
    API_KEY = "seU99BIqWSVbtZ8PmW0PTnNSLpWsj8WE43JFKwzLHPGu7Wb4ZFwE6fjddljcGK87"
    API_SECRET = "0snc4bvVMlK0OlSahiW0grsMrRAzapDj17J99gGxokes1LRZyi2NEs9n4vMJ6iVx"
    
"""