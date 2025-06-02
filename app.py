import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback
import requests

# Vérification des dépendances
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import requests
except ImportError as e:
    st.error(f"Erreur de dépendance : {e}. Assurez-vous que toutes les dépendances sont listées dans requirements.txt.")
    st.stop()

st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H4*")

# Forex pairs - Limited to major pairs only
FOREX_PAIRS_TD = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD'
]

def ema(s, p): 
    return s.ewm(span=p, adjust=False).mean()

def rma(s, p): 
    return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    try:
        hl = int(p/2)
        sl = int(np.sqrt(p))
        wma1 = dc.rolling(window=hl).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), 
            raw=True
        )
        wma2 = dc.rolling(window=p).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), 
            raw=True
        )
        diff = 2 * wma1 - wma2
        return diff.rolling(window=sl).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), 
            raw=True
        )
    except Exception as e:
        print(f"Erreur Hull MA: {traceback.format_exc()}")
        return pd.Series([np.nan] * len(dc), index=dc.index)

def rsi_pine(po4, p=10): 
    try:
        d = po4.diff()
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)
        rs = ag / al.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        print(f"Erreur RSI: {traceback.format_exc()}")
        return pd.Series([50] * len(po4), index=po4.index)

def adx_pine(h, l, c, p=14):
    try:
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rma(tr, p)
        
        um = h.diff()
        dm = l.shift(1) - l
        
        pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=h.index)
        
        satr = atr.replace(0, 1e-9)
        pdi = 100 * (rma(pdm, p) / satr)
        mdi = 100 * (rma(mdm, p) / satr)
        
        dxden = (pdi + mdi).replace(0, 1e-9)
        dx = 100 * (abs(pdi - mdi) / dxden)
        return rma(dx, p).fillna(0)
    except Exception as e:
        print(f"Erreur ADX: {traceback.format_exc()}")
        return pd.Series([0] * len(h), index=h.index)

def heiken_ashi_pine(dfo):
    try:
        ha = pd.DataFrame(index=dfo.index)
        if dfo.empty:
            ha['HA_Open'] = pd.Series(dtype=float)
            ha['HA_Close'] = pd.Series(dtype=float)
            return ha['HA_Open'], ha['HA_Close']
        
        ha['HA_Close'] = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
        ha['HA_Open'] = np.nan
        
        if not dfo.empty:
            ha.iloc[0, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
            for i in range(1, len(dfo)):
                ha.iloc[i, ha.columns.get_loc('HA_Open')] = (
                    ha.iloc[i-1, ha.columns.get_loc('HA_Open')] + 
                    ha.iloc[i-1, ha.columns.get_loc('HA_Close')]
                ) / 2
        
        return ha['HA_Open'], ha['HA_Close']
    except Exception as e:
        print(f"Erreur Heiken Ashi: {traceback.format_exc()}")
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)
        
        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
        hao_i, hac_i = heiken_ashi_pine(hai)
        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)
        return sho, shc
    except Exception as e:
        print(f"Erreur Smoothed Heiken Ashi: {traceback.format_exc()}")
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if len(df_high) < min_len_req or len(df_low) < min_len_req or len(df_close) < min_len_req:
            print(f"Ichi: Data < ({len(df_close)}) vs req {min_len_req}.")
            return 0
        
        ts = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        ks = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min())
   
                     
