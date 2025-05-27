from mcp.server.fastmcp import FastMCP
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

mcp = FastMCP("GlobalStockInvestor", dependencies=["requests", "pandas", "tabulate", "numpy", "scipy"])

API_KEY = "Aplha Vantage API Key" 

@dataclass
class MarketData:
    symbol: str
    interval: str
    data: pd.DataFrame
    last_updated: datetime
    
@dataclass
class FractalPattern:
    level: int
    pattern_type: str  # 'support', 'resistance', 'reversal'
    price: float
    date: str
    strength: float
    confluence_score: int

class AlphaVantageAPI:
    @staticmethod
    def get_daily_data(symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Fetch daily data from AlphaVantage API for long-term analysis"""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Note" in data:
            print(f"API Note: {data['Note']}")
            
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            raise ValueError(f"No daily time series data found for {symbol}")
            
        time_series = data[time_series_key]
        
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        df.columns = [col.split(". ")[1] for col in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df

    @staticmethod
    def get_weekly_data(symbol: str) -> pd.DataFrame:
        """Fetch weekly data from AlphaVantage API"""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
            
        time_series_key = "Weekly Time Series"
        if time_series_key not in data:
            raise ValueError(f"No weekly time series data found for {symbol}")
            
        time_series = data[time_series_key]
        
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        df.columns = [col.split(". ")[1] for col in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df

    @staticmethod
    def get_company_overview(symbol: str) -> Dict[str, Any]:
        """Fetch company fundamental data"""
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
            
        return data

market_data_cache: Dict[str, MarketData] = {}

# Stock symbols mapping for different markets
INDIAN_STOCKS = {
    "RELIANCE": "RELIANCE.BSE",
    "TCS": "TCS.BSE", 
    "INFY": "INFY.BSE",
    "HDFCBANK": "HDFCBANK.BSE",
    "ICICIBANK": "ICICIBANK.BSE",
    "SBIN": "SBIN.BSE",
    "BHARTIARTL": "BHARTIARTL.BSE",
    "ITC": "ITC.BSE",
    "LT": "LT.BSE",
    "KOTAKBANK": "KOTAKBANK.BSE",
    "IDBI": "IDBI.BO",
    "WIPRO": "WIPRO.BSE",
    "MARUTI": "MARUTI.BSE",
    "HINDUNILVR": "HINDUNILVR.BSE",
    "BAJFINANCE": "BAJFINANCE.BSE",
    "ASIANPAINT": "ASIANPAINT.BSE",
    "NESTLEIND": "NESTLEIND.BSE",
    "TITAN": "TITAN.BSE",
    "ULTRACEMCO": "ULTRACEMCO.BSE",
    "POWERGRID": "POWERGRID.BSE"
}

# Common US stocks (no suffix needed)
US_STOCKS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "BRK.A",
    "V", "JNJ", "WMT", "JPM", "PG", "UNH", "MA", "HD", "NFLX", "DIS", "ADBE", "CRM",
    "PYPL", "INTC", "CMCSA", "VZ", "KO", "PFE", "NKE", "MRK", "ABT", "TMO", "COST",
    "AVGO", "PEP", "ORCL", "XOM", "CVX", "WFC", "BAC", "ABBV", "AMD", "COP", "LLY"
}

def detect_stock_market(symbol: str) -> tuple[str, str]:
    """
    Detect if a stock is Indian or US and return formatted symbol and market
    
    Args:
        symbol: Raw stock symbol
        
    Returns:
        Tuple of (formatted_symbol, market_type)
    """
    symbol_upper = symbol.upper().strip()
    
    # Check if it's a predefined Indian stock
    if symbol_upper in INDIAN_STOCKS:
        return INDIAN_STOCKS[symbol_upper], "INDIAN"
    
    # Check if it's already formatted for Indian markets
    if any(suffix in symbol_upper for suffix in ['.BSE', '.NS', '.BO']):
        return symbol_upper, "INDIAN"
    
    # Check if it's a known US stock
    if symbol_upper in US_STOCKS:
        return symbol_upper, "US"
    
    # Check if it looks like a US stock (no dots, common patterns)
    if '.' not in symbol_upper and len(symbol_upper) <= 5:
        # Try as US stock first
        return symbol_upper, "US"
    
    # Default: try as Indian stock with .BSE suffix
    return f"{symbol_upper}.BSE", "INDIAN"

def get_currency_symbol(market_type: str) -> str:
    """Get appropriate currency symbol based on market"""
    return "₹" if market_type == "INDIAN" else "$"

class FractalAnalyzer:
    """Advanced fractal analysis for market structure identification"""
    
    @staticmethod
    def identify_fractal_points(data: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
        """
        Identify fractal high and low points using local extrema
        
        Args:
            data: DataFrame with OHLC data
            window: Window size for fractal identification
            
        Returns:
            Dictionary with fractal highs and lows indices
        """
        # Find local maxima (fractal highs)
        highs = argrelextrema(data['high'].values, np.greater, order=window)[0]
        
        # Find local minima (fractal lows)
        lows = argrelextrema(data['low'].values, np.less, order=window)[0]
        
        return {
            'fractal_highs': highs.tolist(),
            'fractal_lows': lows.tolist()
        }
    
    @staticmethod
    def calculate_fractal_levels(data: pd.DataFrame, fractals: Dict[str, List[int]]) -> List[FractalPattern]:
        """
        Calculate fractal support and resistance levels with strength scoring
        
        Args:
            data: DataFrame with OHLC data
            fractals: Dictionary with fractal points
            
        Returns:
            List of FractalPattern objects
        """
        patterns = []
        current_price = data['close'].iloc[-1]
        
        # Process fractal highs (resistance levels)
        for idx in fractals['fractal_highs']:
            if idx < len(data):
                price = data['high'].iloc[idx]
                date = data.index[idx].strftime('%Y-%m-%d')
                
                # Calculate strength based on volume and time
                volume_strength = data['volume'].iloc[idx] / data['volume'].rolling(20).mean().iloc[idx] if 'volume' in data.columns else 1.0
                time_strength = max(0.1, 1.0 - (len(data) - idx) / len(data))  # Recent fractals are stronger
                
                # Price proximity to current levels
                price_proximity = abs(price - current_price) / current_price
                proximity_strength = max(0.1, 1.0 - price_proximity * 2)
                
                strength = (volume_strength + time_strength + proximity_strength) / 3
                
                # Count confluence (how many times price tested this level)
                confluence_range = price * 0.02  # 2% range
                confluence_count = len([p for p in data['high'] if abs(p - price) <= confluence_range])
                
                pattern = FractalPattern(
                    level=1,
                    pattern_type='resistance',
                    price=price,
                    date=date,
                    strength=min(strength, 1.0),
                    confluence_score=confluence_count
                )
                patterns.append(pattern)
        
        # Process fractal lows (support levels)
        for idx in fractals['fractal_lows']:
            if idx < len(data):
                price = data['low'].iloc[idx]
                date = data.index[idx].strftime('%Y-%m-%d')
                
                # Calculate strength
                volume_strength = data['volume'].iloc[idx] / data['volume'].rolling(20).mean().iloc[idx] if 'volume' in data.columns else 1.0
                time_strength = max(0.1, 1.0 - (len(data) - idx) / len(data))
                
                price_proximity = abs(price - current_price) / current_price
                proximity_strength = max(0.1, 1.0 - price_proximity * 2)
                
                strength = (volume_strength + time_strength + proximity_strength) / 3
                
                # Count confluence
                confluence_range = price * 0.02
                confluence_count = len([p for p in data['low'] if abs(p - price) <= confluence_range])
                
                pattern = FractalPattern(
                    level=1,
                    pattern_type='support',
                    price=price,
                    date=date,
                    strength=min(strength, 1.0),
                    confluence_score=confluence_count
                )
                patterns.append(pattern)
        
        return patterns
    
    @staticmethod
    def identify_multi_timeframe_structure(daily_data: pd.DataFrame, weekly_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market structure across multiple timeframes
        
        Args:
            daily_data: Daily OHLC data
            weekly_data: Weekly OHLC data
            
        Returns:
            Multi-timeframe structure analysis
        """
        # Analyze daily fractals
        daily_fractals = FractalAnalyzer.identify_fractal_points(daily_data, window=3)
        daily_patterns = FractalAnalyzer.calculate_fractal_levels(daily_data, daily_fractals)
        
        # Analyze weekly fractals
        weekly_fractals = FractalAnalyzer.identify_fractal_points(weekly_data, window=2)
        weekly_patterns = FractalAnalyzer.calculate_fractal_levels(weekly_data, weekly_fractals)
        
        current_price = daily_data['close'].iloc[-1]
        
        # Find strongest levels
        all_patterns = daily_patterns + weekly_patterns
        support_levels = [p for p in all_patterns if p.pattern_type == 'support' and p.price < current_price]
        resistance_levels = [p for p in all_patterns if p.pattern_type == 'resistance' and p.price > current_price]
        
        # Sort by strength and proximity
        support_levels.sort(key=lambda x: (x.strength * x.confluence_score, -abs(x.price - current_price)), reverse=True)
        resistance_levels.sort(key=lambda x: (x.strength * x.confluence_score, -abs(x.price - current_price)), reverse=True)
        
        # Identify market structure
        structure = FractalAnalyzer.determine_market_structure(daily_data, daily_patterns)
        
        return {
            'current_price': current_price,
            'market_structure': structure,
            'key_support_levels': support_levels[:5],  # Top 5 support levels
            'key_resistance_levels': resistance_levels[:5],  # Top 5 resistance levels
            'daily_fractals_count': len(daily_patterns),
            'weekly_fractals_count': len(weekly_patterns),
            'structure_strength': FractalAnalyzer.calculate_structure_strength(all_patterns, current_price)
        }
    
    @staticmethod
    def determine_market_structure(data: pd.DataFrame, patterns: List[FractalPattern]) -> str:
        """
        Determine overall market structure based on fractal patterns
        
        Args:
            data: OHLC data
            patterns: List of fractal patterns
            
        Returns:
            Market structure description
        """
        current_price = data['close'].iloc[-1]
        recent_data = data.tail(50)  # Last 50 days
        
        # Analyze recent highs and lows pattern
        recent_highs = [p for p in patterns if p.pattern_type == 'resistance' and 
                       pd.to_datetime(p.date) >= recent_data.index.min()]
        recent_lows = [p for p in patterns if p.pattern_type == 'support' and 
                      pd.to_datetime(p.date) >= recent_data.index.min()]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for higher highs and higher lows (uptrend)
            highs_ascending = all(recent_highs[i].price < recent_highs[i+1].price 
                                for i in range(len(recent_highs)-1))
            lows_ascending = all(recent_lows[i].price < recent_lows[i+1].price 
                               for i in range(len(recent_lows)-1))
            
            # Check for lower highs and lower lows (downtrend)
            highs_descending = all(recent_highs[i].price > recent_highs[i+1].price 
                                 for i in range(len(recent_highs)-1))
            lows_descending = all(recent_lows[i].price > recent_lows[i+1].price 
                                for i in range(len(recent_lows)-1))
            
            if highs_ascending and lows_ascending:
                return "STRONG UPTREND - Higher Highs & Higher Lows"
            elif highs_descending and lows_descending:
                return "STRONG DOWNTREND - Lower Highs & Lower Lows"
            elif highs_ascending and not lows_descending:
                return "BULLISH STRUCTURE - Higher Highs with Stable Lows"
            elif lows_descending and not highs_ascending:
                return "BEARISH STRUCTURE - Lower Lows with Stable Highs"
            else:
                return "CONSOLIDATION - Mixed Structure"
        
        # Fallback to simple trend analysis
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        sma_50 = data['close'].rolling(50).mean().iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "BULLISH TREND - Price Above Moving Averages"
        elif current_price < sma_20 < sma_50:
            return "BEARISH TREND - Price Below Moving Averages"
        else:
            return "SIDEWAYS MARKET - Mixed Signals"
    
    @staticmethod
    def calculate_structure_strength(patterns: List[FractalPattern], current_price: float) -> Dict[str, float]:
        """
        Calculate the strength of current market structure
        
        Args:
            patterns: List of fractal patterns
            current_price: Current market price
            
        Returns:
            Dictionary with structure strength metrics
        """
        if not patterns:
            return {'support_strength': 0, 'resistance_strength': 0, 'overall_strength': 0}
        
        # Calculate support strength
        support_patterns = [p for p in patterns if p.pattern_type == 'support' and p.price < current_price]
        support_strength = sum(p.strength * p.confluence_score for p in support_patterns[:3]) / 3 if support_patterns else 0
        
        # Calculate resistance strength
        resistance_patterns = [p for p in patterns if p.pattern_type == 'resistance' and p.price > current_price]
        resistance_strength = sum(p.strength * p.confluence_score for p in resistance_patterns[:3]) / 3 if resistance_patterns else 0
        
        # Overall structure strength
        overall_strength = (support_strength + resistance_strength) / 2
        
        return {
            'support_strength': min(support_strength, 10.0),
            'resistance_strength': min(resistance_strength, 10.0),
            'overall_strength': min(overall_strength, 10.0)
        }

@mcp.resource("config://supported-markets")
def get_supported_markets() -> str:
    """Supported markets and popular stocks"""
    return f"""
    Indian Stocks: {list(INDIAN_STOCKS.keys())}
    US Stocks: {list(US_STOCKS)}
    
    Market Detection:
    - Symbols ending with .BSE, .NS, .BO are treated as Indian
    - Common US ticker symbols are auto-detected
    - Unknown symbols default to US first, then Indian (.BSE)
    """

@mcp.tool()
def fractal_market_analysis(symbol: str, analysis_period: int = 252) -> Dict[str, Any]:
    """
    Perform advanced fractal-based market structure analysis
    
    Args:
        symbol: The ticker symbol to analyze (US or Indian stock)
        analysis_period: Number of days to analyze (default 252 = 1 year)
        
    Returns:
        Dictionary with comprehensive fractal analysis
    """
    # Detect market and format symbol
    formatted_symbol, market_type = detect_stock_market(symbol)
    currency = get_currency_symbol(market_type)
    
    try:
        # Get daily and weekly data
        daily_cache_key = f"{formatted_symbol}_daily"
        weekly_cache_key = f"{formatted_symbol}_weekly"
        
        if daily_cache_key not in market_data_cache:
            daily_df = AlphaVantageAPI.get_daily_data(formatted_symbol, outputsize="full")
            market_data_cache[daily_cache_key] = MarketData(
                symbol=formatted_symbol,
                interval="daily",
                data=daily_df,
                last_updated=datetime.now()
            )
        
        if weekly_cache_key not in market_data_cache:
            weekly_df = AlphaVantageAPI.get_weekly_data(formatted_symbol)
            market_data_cache[weekly_cache_key] = MarketData(
                symbol=formatted_symbol,
                interval="weekly", 
                data=weekly_df,
                last_updated=datetime.now()
            )
        
        daily_data = market_data_cache[daily_cache_key].data.tail(analysis_period).copy()
        weekly_data = market_data_cache[weekly_cache_key].data.copy()
        
        # Perform multi-timeframe fractal analysis
        structure_analysis = FractalAnalyzer.identify_multi_timeframe_structure(daily_data, weekly_data)
        
        current_price = structure_analysis['current_price']
        
        # Calculate entry/exit zones based on fractals
        key_support = structure_analysis['key_support_levels']
        key_resistance = structure_analysis['key_resistance_levels']
        
        # Determine optimal entry zones
        if key_support:
            nearest_support = key_support[0]
            support_zone_lower = nearest_support.price * 0.98  # 2% below support
            support_zone_upper = nearest_support.price * 1.02  # 2% above support
        else:
            support_zone_lower = current_price * 0.95
            support_zone_upper = current_price * 0.97
        
        # Determine target zones
        if key_resistance:
            nearest_resistance = key_resistance[0]
            target_zone_lower = nearest_resistance.price * 0.98
            target_zone_upper = nearest_resistance.price * 1.02
        else:
            target_zone_lower = current_price * 1.05
            target_zone_upper = current_price * 1.08
        
        # Calculate risk-reward ratio
        entry_price = (support_zone_lower + support_zone_upper) / 2
        target_price = (target_zone_lower + target_zone_upper) / 2
        stop_loss = support_zone_lower * 0.97  # 3% below entry zone
        
        potential_profit = target_price - entry_price
        potential_loss = entry_price - stop_loss
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        
        # Generate trading signals based on fractal structure
        signals = []
        
        if current_price <= support_zone_upper and current_price >= support_zone_lower:
            signals.append("BUY ZONE - Price near key support level")
        elif current_price >= target_zone_lower:
            signals.append("PROFIT TAKING ZONE - Price near resistance")
        elif structure_analysis['market_structure'].startswith('STRONG UPTREND'):
            signals.append("TREND FOLLOWING - Look for pullback entries")
        elif structure_analysis['market_structure'].startswith('STRONG DOWNTREND'):
            signals.append("AVOID LONGS - Consider shorting opportunities")
        
        # Format support and resistance levels for display
        support_levels_info = []
        for i, level in enumerate(key_support[:3]):
            support_levels_info.append({
                'rank': i + 1,
                'price': level.price,
                'strength': level.strength,
                'confluence': level.confluence_score,
                'date': level.date,
                'distance_percent': ((current_price - level.price) / current_price) * 100
            })
        
        resistance_levels_info = []
        for i, level in enumerate(key_resistance[:3]):
            resistance_levels_info.append({
                'rank': i + 1,
                'price': level.price,
                'strength': level.strength,
                'confluence': level.confluence_score,
                'date': level.date,
                'distance_percent': ((level.price - current_price) / current_price) * 100
            })
        
        return {
            "symbol": formatted_symbol,
            "market_type": market_type,
            "currency": currency,
            "current_price": current_price,
            "market_structure": structure_analysis['market_structure'],
            "structure_strength": structure_analysis['structure_strength'],
            "key_support_levels": support_levels_info,
            "key_resistance_levels": resistance_levels_info,
            "entry_zone": {
                "lower": support_zone_lower,
                "upper": support_zone_upper,
                "optimal": entry_price
            },
            "target_zone": {
                "lower": target_zone_lower,
                "upper": target_zone_upper,
                "optimal": target_price
            },
            "stop_loss": stop_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "trading_signals": signals,
            "fractal_summary": {
                "daily_patterns": structure_analysis['daily_fractals_count'],
                "weekly_patterns": structure_analysis['weekly_fractals_count'],
                "analysis_period_days": analysis_period
            }
        }
        
    except Exception as e:
        # Try alternative symbol format if first attempt fails
        if market_type == "US":
            try:
                indian_symbol = f"{symbol.upper()}.BSE"
                # Recursive call with Indian format
                return fractal_market_analysis(indian_symbol, analysis_period)
            except:
                pass
        
        return {
            "symbol": formatted_symbol,
            "market_type": market_type,
            "error": f"Unable to perform fractal analysis: {str(e)}",
            "current_price": 0,
            "market_structure": "UNKNOWN",
            "trading_signals": ["Unable to analyze - check symbol or try again later"]
        }

@mcp.tool()
def calculate_long_term_indicators(symbol: str, short_period: int = 50, long_period: int = 200) -> Dict[str, Any]:
    """
    Calculate long-term moving averages and trend analysis for any stock
    
    Args:
        symbol: The ticker symbol to analyze (US or Indian stock)
        short_period: Short moving average period in days (default 50)
        long_period: Long moving average period in days (default 200)
        
    Returns:
        Dictionary with long-term trend analysis
    """
    # Detect market and format symbol
    formatted_symbol, market_type = detect_stock_market(symbol)
    currency = get_currency_symbol(market_type)
    
    cache_key = f"{formatted_symbol}_daily"
    
    if cache_key not in market_data_cache:
        try:
            df = AlphaVantageAPI.get_daily_data(formatted_symbol, outputsize="full")
            market_data_cache[cache_key] = MarketData(
                symbol=formatted_symbol,
                interval="daily",
                data=df,
                last_updated=datetime.now()
            )
        except Exception as e:
            # If US stock fails, try as Indian stock
            if market_type == "US":
                try:
                    indian_symbol = f"{symbol.upper()}.BSE"
                    df = AlphaVantageAPI.get_daily_data(indian_symbol, outputsize="full")
                    formatted_symbol = indian_symbol
                    market_type = "INDIAN"
                    currency = "₹"
                    cache_key = f"{formatted_symbol}_daily"
                    market_data_cache[cache_key] = MarketData(
                        symbol=formatted_symbol,
                        interval="daily",
                        data=df,
                        last_updated=datetime.now()
                    )
                except:
                    raise e
            else:
                raise e
    
    data = market_data_cache[cache_key].data.copy()
    
    # Calculate moving averages
    data[f'SMA{short_period}'] = data['close'].rolling(window=short_period).mean()
    data[f'SMA{long_period}'] = data['close'].rolling(window=long_period).mean()
    
    # Calculate price performance
    data['returns_1m'] = data['close'].pct_change(21)  # 1 month
    data['returns_3m'] = data['close'].pct_change(63)  # 3 months
    data['returns_6m'] = data['close'].pct_change(126) # 6 months
    data['returns_1y'] = data['close'].pct_change(252) # 1 year
    
    # Get latest values
    latest = data.iloc[-1]
    current_price = latest['close']
    short_ma = latest[f'SMA{short_period}']
    long_ma = latest[f'SMA{long_period}']
    
    # Calculate volatility (standard deviation of daily returns)
    daily_returns = data['close'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
    
    # Determine long-term trend
    if current_price > short_ma > long_ma:
        trend = "STRONG UPTREND"
    elif current_price > short_ma and short_ma < long_ma:
        trend = "RECOVERY PHASE"
    elif current_price < short_ma < long_ma:
        trend = "STRONG DOWNTREND"
    elif current_price < short_ma and short_ma > long_ma:
        trend = "CORRECTION PHASE"
    else:
        trend = "SIDEWAYS/CONSOLIDATION"
    
    # Support and resistance levels
    recent_data = data.tail(252)  # Last 1 year
    resistance = recent_data['high'].max()
    support = recent_data['low'].min()
    
    return {
        "symbol": formatted_symbol,
        "market_type": market_type,
        "currency": currency,
        "current_price": current_price,
        f"SMA{short_period}": short_ma,
        f"SMA{long_period}": long_ma,
        "trend": trend,
        "volatility": volatility,
        "returns_1m": latest['returns_1m'] if pd.notna(latest['returns_1m']) else 0,
        "returns_3m": latest['returns_3m'] if pd.notna(latest['returns_3m']) else 0,
        "returns_6m": latest['returns_6m'] if pd.notna(latest['returns_6m']) else 0,
        "returns_1y": latest['returns_1y'] if pd.notna(latest['returns_1y']) else 0,
        "support_level": support,
        "resistance_level": resistance,
        "distance_from_52w_high": (current_price - resistance) / resistance * 100,
        "distance_from_52w_low": (current_price - support) / support * 100
    }

@mcp.tool()
def calculate_value_metrics(symbol: str) -> Dict[str, Any]:
    """
    Calculate value investing metrics using fundamental data for any stock
    
    Args:
        symbol: The ticker symbol to analyze (US or Indian)
        
    Returns:
        Dictionary with valuation metrics
    """
    # Detect market and format symbol
    formatted_symbol, market_type = detect_stock_market(symbol)
    
    try:
        overview = AlphaVantageAPI.get_company_overview(formatted_symbol)
        
        # Extract key metrics
        pe_ratio = float(overview.get('PERatio', 0)) if overview.get('PERatio') not in ['None', 'N/A', ''] else 0
        pb_ratio = float(overview.get('PriceToBookRatio', 0)) if overview.get('PriceToBookRatio') not in ['None', 'N/A', ''] else 0
        dividend_yield = float(overview.get('DividendYield', 0)) if overview.get('DividendYield') not in ['None', 'N/A', ''] else 0
        roe = float(overview.get('ReturnOnEquityTTM', 0)) if overview.get('ReturnOnEquityTTM') not in ['None', 'N/A', ''] else 0
        debt_to_equity = float(overview.get('DebtToEquityRatio', 0)) if overview.get('DebtToEquityRatio') not in ['None', 'N/A', ''] else 0
        
        # Value assessment (adjusted for market differences)
        value_score = 0
        
        # PE Ratio scoring (different thresholds for US vs Indian markets)
        if market_type == "US":
            if 0 < pe_ratio < 20:
                value_score += 2
            elif 20 <= pe_ratio < 30:
                value_score += 1
        else:  # Indian market
            if 0 < pe_ratio < 15:
                value_score += 2
            elif 15 <= pe_ratio < 25:
                value_score += 1
        
        # PB Ratio scoring
        if 0 < pb_ratio < 1.5:
            value_score += 2
        elif 1.5 <= pb_ratio < 3:
            value_score += 1
            
        # Dividend yield scoring
        if dividend_yield > 0.02:  # 2%+
            value_score += 1
            
        # ROE scoring
        if roe > 0.15:  # 15%+
            value_score += 1
            
        # Debt to equity scoring
        if debt_to_equity < 0.5:
            value_score += 1
        
        return {
            "symbol": formatted_symbol,
            "market_type": market_type,
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "dividend_yield": dividend_yield * 100 if dividend_yield else 0,  # Convert to percentage
            "roe": roe * 100 if roe else 0,  # Convert to percentage
            "debt_to_equity": debt_to_equity,
            "market_cap": overview.get('MarketCapitalization', 'N/A'),
            "sector": overview.get('Sector', 'N/A'),
            "industry": overview.get('Industry', 'N/A'),
            "country": overview.get('Country', 'N/A'),
            "exchange": overview.get('Exchange', 'N/A'),
            "value_score": value_score,
            "max_value_score": 7
        }
    except Exception as e:
        # Try alternative symbol format if first attempt fails
        if market_type == "US":
            try:
                indian_symbol = f"{symbol.upper()}.BSE"
                overview = AlphaVantageAPI.get_company_overview(indian_symbol)
                formatted_symbol = indian_symbol
                market_type = "INDIAN"
                # Repeat the same logic above with the new symbol
                # (code would be duplicated here for brevity)
            except:
                pass
                
        return {
            "symbol": formatted_symbol,
            "market_type": market_type,
            "error": f"Unable to fetch fundamental data: {str(e)}",
            "pe_ratio": 0,
            "pb_ratio": 0,
            "dividend_yield": 0,
            "roe": 0,
            "debt_to_equity": 0,
            "market_cap": "N/A",
            "sector": "N/A",
            "value_score": 1,
            "max_value_score": 7
        }

@mcp.tool()
def investment_recommendation(symbol: str) -> Dict[str, Any]:
    """
    Provide long-term investment recommendation based on technical and fundamental analysis
    
    Args:
        symbol: The ticker symbol to analyze (US or Indian)
        
    Returns:
        Dictionary with investment recommendation
    """
    # Get technical analysis
    technical = calculate_long_term_indicators(symbol)
    
    # Get fundamental analysis
    fundamental = calculate_value_metrics(symbol)
    
    market_type = technical["market_type"]
    currency = technical["currency"]
    
    # Calculate investment score
    investment_score = 0
    
    # Technical scoring
    if "UPTREND" in technical["trend"]:
        investment_score += 2
    elif "RECOVERY" in technical["trend"]:
        investment_score += 1
    elif "DOWNTREND" in technical["trend"]:
        investment_score -= 2
    elif "CORRECTION" in technical["trend"]:
        investment_score -= 1
    
    # Performance scoring (adjusted for market type)
    annual_return_threshold = 0.15 if market_type == "INDIAN" else 0.12  # Higher expectations for Indian market
    if technical["returns_1y"] > annual_return_threshold:
        investment_score += 2
    elif technical["returns_1y"] > 0:
        investment_score += 1
    elif technical["returns_1y"] < -annual_return_threshold:
        investment_score -= 2
    elif technical["returns_1y"] < 0:
        investment_score -= 1
    
    # Volatility scoring (adjusted for market type)
    low_vol_threshold = 0.25 if market_type == "US" else 0.35  # Higher volatility tolerance for Indian stocks
    high_vol_threshold = 0.4 if market_type == "US" else 0.6
    
    if technical["volatility"] < low_vol_threshold:
        investment_score += 1
    elif technical["volatility"] > high_vol_threshold:
        investment_score -= 1
    
    # Value scoring
    investment_score += fundamental["value_score"] - 3  # Normalize around 0
    
    # Determine recommendation
    if investment_score >= 4:
        recommendation = "STRONG BUY"
        action = "Consider accumulating on any weakness"
    elif investment_score >= 2:
        recommendation = "BUY"
        action = "Good long-term investment opportunity"
    elif investment_score >= 0:
        recommendation = "HOLD"
        action = "Maintain existing position, monitor quarterly results"
    elif investment_score >= -2:
        recommendation = "WEAK HOLD"
        action = "Consider reducing position size"
    else:
        recommendation = "AVOID"
        action = "Look for better investment opportunities"
    
    # Risk assessment (market-adjusted)
    high_vol_threshold = 0.4 if market_type == "US" else 0.5
    if technical["volatility"] > high_vol_threshold or technical["distance_from_52w_high"] < -30:
        risk_level = "HIGH"
    elif technical["volatility"] < (0.25 if market_type == "US" else 0.3) and abs(technical["distance_from_52w_high"]) < 20:
        risk_level = "LOW"
    else:
        risk_level = "MEDIUM"
    
    market_context = "Indian markets" if market_type == "INDIAN" else "US markets"
    
    analysis = f"""# Long-term Investment Analysis for {technical["symbol"]}

## Investment Recommendation: {recommendation}
**Market:** {market_type} | **Risk Level:** {risk_level} | **Investment Score:** {investment_score}/10

## Technical Analysis
- **Current Trend:** {technical["trend"]}
- **Current Price:** {currency}{technical["current_price"]:.2f}
- **50-day SMA:** {currency}{technical["SMA50"]:.2f}
- **200-day SMA:** {currency}{technical["SMA200"]:.2f}
- **Annual Volatility:** {technical["volatility"]*100:.1f}%

## Performance Metrics
- **1 Month Return:** {technical["returns_1m"]*100:.1f}%
- **3 Month Return:** {technical["returns_3m"]*100:.1f}%
- **6 Month Return:** {technical["returns_6m"]*100:.1f}%
- **1 Year Return:** {technical["returns_1y"]*100:.1f}%

## Valuation Metrics
- **P/E Ratio:** {fundamental["pe_ratio"]:.1f}
- **P/B Ratio:** {fundamental["pb_ratio"]:.1f}
- **Dividend Yield:** {fundamental["dividend_yield"]:.1f}%
- **ROE:** {fundamental["roe"]:.1f}%
- **Debt/Equity:** {fundamental["debt_to_equity"]:.2f}
- **Sector:** {fundamental.get("sector", "N/A")}
- **Value Score:** {fundamental["value_score"]}/{fundamental.get("max_value_score", 7)}

## Key Levels
- **52-week High:** {currency}{technical["resistance_level"]:.2f} ({technical["distance_from_52w_high"]:.1f}% from current)
- **52-week Low:** {currency}{technical["support_level"]:.2f} ({technical["distance_from_52w_low"]:.1f}% from current)

## Investment Strategy
{action}

## Risk Factors to Monitor
- Quarterly earnings growth
- Sector-specific headwinds
- Market volatility and corrections
- Regulatory changes in {market_context}
{"- Currency fluctuations (USD/INR)" if market_type == "INDIAN" else "- Interest rate changes and Fed policy"}
"""
    
    return {
        "symbol": technical["symbol"],
        "market_type": market_type,
        "recommendation": recommendation,
        "risk_level": risk_level,
        "investment_score": investment_score,
        "current_price": technical["current_price"],
        "trend": technical["trend"],
        "annual_return": technical["returns_1y"],
        "pe_ratio": fundamental["pe_ratio"],
        "dividend_yield": fundamental["dividend_yield"],
        "sector": fundamental.get("sector", "N/A"),
        "analysis": analysis
    }

@mcp.tool()
def portfolio_diversification_check(symbols: List[str]) -> Dict[str, Any]:
    """
    Analyze portfolio diversification across sectors and markets
    
    Args:
        symbols: List of ticker symbols in portfolio
        
    Returns:
        Dictionary with diversification analysis
    """
    sector_allocation = {}
    market_allocation = {"US": 0, "INDIAN": 0}
    total_positions = len(symbols)
    
    for symbol in symbols:
        try:
            fundamental = calculate_value_metrics(symbol)
            sector = fundamental.get("sector", "Unknown")
            market = fundamental.get("market_type", "Unknown")
            
            # Sector allocation
            if sector in sector_allocation:
                sector_allocation[sector] += 1
            else:
                sector_allocation[sector] = 1
                
            # Market allocation
            if market in market_allocation:
                market_allocation[market] += 1
        except:
            if "Unknown" in sector_allocation:
                sector_allocation["Unknown"] += 1
            else:
                sector_allocation["Unknown"] = 1
    
    # Calculate percentages
    sector_percentages = {sector: (count/total_positions)*100 
                         for sector, count in sector_allocation.items()}
    market_percentages = {market: (count/total_positions)*100 
                         for market, count in market_allocation.items()}
    
    # Diversification score
    max_sector_concentration = max(sector_percentages.values()) if sector_percentages else 0
    market_diversification_bonus = 10 if len([v for v in market_percentages.values() if v > 0]) > 1 else 0
    
    diversification_score = max(0, 100 - max_sector_concentration + market_diversification_bonus)
    
    return {
        "total_positions": total_positions,
        "sector_allocation": sector_allocation,
        "sector_percentages": sector_percentages,
        "market_allocation": market_allocation,
        "market_percentages": market_percentages,
        "diversification_score": diversification_score,
        "recommendation": "EXCELLENT" if diversification_score > 80 else
                         "GOOD" if diversification_score > 70 else 
                         "MODERATE" if diversification_score > 50 else "POOR"
    }

# Enhanced investment-focused prompts
@mcp.prompt()
def analyze_any_stock(symbol: str) -> str:
    """
    Comprehensive long-term investment analysis for any stock (US or Indian)
    """
    return f"""You are a professional investment analyst with expertise in both US and Indian stock markets. 

Please provide a comprehensive long-term investment analysis for {symbol}. The analysis should:

1. Use the investment_recommendation tool to get the overall investment thesis
2. Analyze the technical trend using calculate_long_term_indicators
3. Evaluate the fundamental metrics using calculate_value_metrics
4. **NEW: Perform advanced fractal analysis using fractal_market_analysis for precise entry/exit levels**
5. Consider market-specific factors:
   
   **For US Stocks:**
   - Fed policy and interest rate environment
   - Sector rotation in US markets
   - Dollar strength/weakness impact
   - Regulatory environment (SEC, antitrust)
   
   **For Indian Stocks:**
   - RBI monetary policy
   - Rupee depreciation/appreciation trends
   - Government policy impacts
   - Monsoon and commodity price effects

Provide actionable insights for a long-term investor with a 3-5 year investment horizon. Include specific entry strategies, position sizing recommendations, and risk management approach based on fractal support/resistance levels.

Structure your analysis for clarity and focus on wealth creation through long-term value investing principles."""

@mcp.prompt()
def fractal_trading_strategy(symbol: str, risk_tolerance: str = "moderate") -> str:
    """
    Create a fractal-based trading strategy with precise entry and exit levels
    
    Args:
        symbol: Stock symbol to analyze
        risk_tolerance: "conservative", "moderate", or "aggressive"
    """
    return f"""You are an expert technical analyst specializing in fractal market structure analysis and precision trading.

Create a comprehensive fractal-based trading strategy for {symbol} with {risk_tolerance} risk tolerance.

Please:

1. **Perform fractal_market_analysis** to identify key market structure levels
2. Use **calculate_long_term_indicators** for trend context
3. Get **calculate_value_metrics** for fundamental backdrop

Based on the fractal analysis, provide:

## Entry Strategy
- Precise entry zones based on fractal support levels
- Multiple entry scenarios (breakout vs. pullback)
- Position sizing based on risk tolerance
- Confirmation signals to look for

## Exit Strategy  
- Target zones based on fractal resistance levels
- Stop-loss placement using fractal structure
- Trailing stop methodology
- Partial profit-taking levels

## Risk Management
- Risk per trade based on fractal levels
- Portfolio heat and position correlation
- Market structure invalidation points
- Emergency exit scenarios

## Market Structure Context
- Multi-timeframe fractal alignment
- Support/resistance confluence zones
- Market phase identification
- Structural strength assessment

Tailor the strategy for {risk_tolerance} risk tolerance:
- **Conservative**: Lower position sizes, wider stops, higher confluence requirements
- **Moderate**: Balanced approach with 1-2% risk per trade
- **Aggressive**: Higher position sizes, tighter stops, faster entries

Focus on high-probability setups with strong fractal structure backing."""

@mcp.prompt()
def build_global_portfolio(symbols: str, investment_amount: float, base_currency: str = "USD") -> str:
    """
    Build a diversified long-term portfolio mixing US and Indian stocks
    
    Args:
        symbols: Comma-separated list of stock symbols (US and Indian)
        investment_amount: Total investment amount
        base_currency: Base currency (USD or INR)
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    return f"""You are a global portfolio manager specializing in US and Indian equity markets for long-term wealth creation.

I want to invest {base_currency} {investment_amount:,.0f} across these stocks: {', '.join(symbol_list)}

Please provide:

1. Individual analysis for each stock using the investment_recommendation tool
2. **Enhanced with fractal_market_analysis** for optimal entry timing
3. Portfolio diversification analysis using portfolio_diversification_check
4. Recommended allocation percentages considering:
   - Risk-adjusted returns
   - Geographic diversification (US vs Indian exposure)
   - Sector diversification across both markets
   - Currency risk management
   - Market cap distribution
   - **Fractal structure strength for timing entries**

5. Create a systematic investment plan including:
   - Initial lump sum allocation vs. staged entries based on fractal levels
   - Monthly SIP/DCA recommendations
   - Rebalancing frequency and triggers
   - Currency hedging strategies (if needed)
   - Exit strategies for each position

6. Risk management framework:
   - Maximum position sizes
   - Geographic allocation limits
   - Currency exposure management
   - Portfolio review schedule
   - **Fractal-based stop loss levels for the portfolio**

Focus on building a globally diversified portfolio suitable for long-term wealth creation with a 5-10 year investment horizon, enhanced with precise fractal-based entry and exit timing."""

@mcp.prompt()
def create_sip_strategy(symbols: str, monthly_amount: float, currency: str = "USD") -> str:
    """
    Build a Systematic Investment Plan (SIP) strategy for multiple stocks
    
    Args:
        symbols: Comma-separated stock symbols
        monthly_amount: Monthly investment amount
        currency: Investment currency (USD or INR)
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    return f"""You are a financial advisor specializing in systematic investment planning across global markets.

I want to start a monthly SIP of {currency} {monthly_amount:,.0f} across these stocks: {', '.join(symbol_list)}.

Please provide:

1. Complete investment analysis for each stock using investment_recommendation
2. **Fractal analysis using fractal_market_analysis** to optimize SIP timing
3. SIP allocation strategy:
   - Percentage allocation per stock
   - Dollar/Rupee cost averaging benefits
   - Optimal investment timing and frequency
   - Target accumulation over 1, 3, 5, 10 years
   - **Fractal-based entry zone targeting for better average prices**

4. Enhanced SIP timing with fractal structure:
   - When to increase SIP amounts (near fractal support)
   - When to pause SIP (at fractal resistance/overextension)
   - Multi-timeframe SIP approach (weekly vs monthly)
   - Opportunistic lump sum additions at key fractal levels

5. Risk management for SIP:
   - When to pause/stop SIP for individual stocks
   - When to increase allocation
   - Rebalancing with SIP approach
   - Currency risk management (for cross-border investments)
   - **Portfolio-level fractal structure monitoring**

6. Tax optimization strategies:
   - Long-term capital gains planning
   - Tax-loss harvesting opportunities
   - Jurisdiction-specific tax considerations

7. Monitoring and review framework:
   - Monthly review checklist with fractal structure updates
   - Quarterly rebalancing triggers
   - Annual strategy review
   - **Fractal structure deterioration warning signs**

Make the strategy practical for retail investors with focus on disciplined long-term wealth creation across global markets, enhanced with fractal-based precision timing."""
