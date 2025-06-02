import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback
import requests

# Configuration de la page Streamlit (doit être la première commande Streamlit)
st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")

# Vérification des dépendances (optionnel si requirements.txt est bien géré)
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import requests
except ImportError as e:
    st.error(f"Erreur de dépendance : {e}. Assurez-vous que toutes les dépendances sont listées dans requirements.txt.")
    st.stop()

st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H4*")

# --- Configuration ---
# Forex pairs - Limited to major pairs for demonstration
FOREX_PAIRS_TD = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD'
]
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
# Essayez de récupérer la clé API. Si elle n'est pas trouvée, affichez une erreur claire.
API_KEY = None
try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except (FileNotFoundError, KeyError): # FileNotFoundError pour les secrets locaux, KeyError pour Streamlit Cloud
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) n'a pas été trouvée dans les secrets Streamlit.")
    st.info("Veuillez ajouter votre clé API aux secrets de votre application Streamlit Cloud. Exemple : TWELVE_DATA_API_KEY = 'votrecléapi'")
    st.stop()

if not API_KEY: # Double vérification au cas où la clé serait vide
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) est vide dans les secrets Streamlit.")
    st.stop()

# --- Fonctions Indicateurs (généralement inchangées, mais vérifions les retours en cas d'erreur) ---
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rma(s, p):
    return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    try:
        if dc.empty or len(dc) < p : # Vérification de longueur minimale
             return pd.Series([np.nan] * len(dc), index=dc.index)
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
    except Exception:
        # st.warning(f"Erreur Hull MA: {traceback.format_exc()}") # Peut être bruyant
        return pd.Series([np.nan] * len(dc), index=dc.index)

def rsi_pine(po4, p=10):
    try:
        if po4.empty or len(po4) < p:
            return pd.Series([50] * len(po4), index=po4.index) # Valeur neutre si pas assez de données
        d = po4.diff()
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)
        rs = ag / al.replace(0, 1e-9) # Éviter division par zéro
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # Remplir les NaN initiaux
    except Exception:
        # st.warning(f"Erreur RSI: {traceback.format_exc()}")
        return pd.Series([50] * len(po4), index=po4.index)

def adx_pine(h, l, c, p=14):
    try:
        if h.empty or len(h) < p + 1: # ADX a besoin d'un peu plus de données pour .shift(1) et rma
            return pd.Series([0] * len(h), index=h.index) # Valeur neutre/faible
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rma(tr, p)

        um = h.diff()
        dm = l.shift(1) - l

        pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=h.index)

        satr = atr.replace(0, 1e-9) # Éviter division par zéro
        pdi = 100 * (rma(pdm, p) / satr)
        mdi = 100 * (rma(mdm, p) / satr)

        dxden = (pdi + mdi).replace(0, 1e-9) # Éviter division par zéro
        dx = 100 * (abs(pdi - mdi) / dxden)
        return rma(dx, p).fillna(0) # Remplir les NaN initiaux
    except Exception:
        # st.warning(f"Erreur ADX: {traceback.format_exc()}")
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
            ha.loc[ha.index[0], 'HA_Open'] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
            for i in range(1, len(dfo)):
                ha.loc[ha.index[i], 'HA_Open'] = (
                    ha.loc[ha.index[i-1], 'HA_Open'] +
                    ha.loc[ha.index[i-1], 'HA_Close']
                ) / 2
        return ha['HA_Open'], ha['HA_Close']
    except Exception:
        # st.warning(f"Erreur Heiken Ashi: {traceback.format_exc()}")
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        if dfo.empty or len(dfo) < max(l1,l2): # Vérification de longueur
            empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
            return empty_series, empty_series

        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)

        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
        hao_i, hac_i = heiken_ashi_pine(hai)
        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)
        return sho, shc
    except Exception:
        # st.warning(f"Erreur Smoothed Heiken Ashi: {traceback.format_exc()}")
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p) + kijun_p # Pour les shifts
        if len(df_high) < min_len_req:
            # st.warning(f"Ichi: Données insuffisantes ({len(df_close)}) vs requis {min_len_req}.")
            return 0 # Signal neutre

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        # Kijun-sen (Base Line)
        kijun_sen = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_p)
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2).shift(kijun_p)
        # Chikou Span (Lagging Span)
        chikou_span = df_close.shift(-kijun_p) # Shifté vers le passé par rapport au prix actuel

        # Dernières valeurs (actuelles pour signaux, sauf pour chikou qui est comparé au prix passé)
        c = df_close.iloc[-1]
        ts = tenkan_sen.iloc[-1]
        ks = kijun_sen.iloc[-1]
        ssa = senkou_span_a.iloc[-1] # Nuage futur au niveau de la bougie actuelle
        ssb = senkou_span_b.iloc[-1] # Nuage futur au niveau de la bougie actuelle
        
        # Pour Chikou, on compare sa valeur actuelle (prix d'il y a 26 périodes)
        # avec le prix d'il y a 26 périodes ET le nuage d'il y a 26 périodes.
        # Cependant, pour un signal simple, on regarde sa position par rapport au prix/nuage "actuel" (décalé).
        # Ici, on va simplifier: on regarde si la Chikou Span (prix d'il y a 26p) est au-dessus/dessous du prix d'il y a 26p.
        # Pour un signal de trading, on regarde où est Chikou par rapport aux prix d'il y a 26 périodes.
        # Pour la "force" du signal actuel, on peut considérer la Chikou actuelle par rapport au nuage actuel (décalé).
        # La chikou span est le prix actuel décalé de kijun_p périodes en arrière.
        # On la compare au prix qui était à ce moment là (il y a kijun_p périodes)
        # Pour le signal: on veut `chikou_span.iloc[-1]` (qui est `df_close.iloc[-1-kijun_p]`)
        # par rapport au nuage qui était `kijun_p` périodes en arrière.
        # Ou, plus classiquement:
        # Le prix actuel `c` doit être au-dessus/dessous du nuage actuel (`ssa`, `ssb`)
        # Le `chikou_span.iloc[-1]` (qui est `df_close.iloc[-1-kijun_p]`)
        # doit être au-dessus/dessous du nuage `kijun_p` périodes dans le passé.

        # Conditions haussières
        bullish_price_above_kumo = c > ssa and c > ssb
        bullish_ts_above_ks = ts > ks
        bullish_chikou_above_price_past = chikou_span.iloc[-1-kijun_p] > df_close.iloc[-1-kijun_p] if len(chikou_span) > kijun_p and len(df_close) > kijun_p else False #Approximation
        # Pour le signal actuel, on vérifie si le cours est au-dessus du nuage, et Tenkan > Kijun
        # et Chikou est au-dessus des prix d'il y a 26 périodes.
        # Et le nuage futur est haussier (SSA > SSB)
        bullish_future_kumo = ssa > ssb


        # Conditions baissières
        bearish_price_below_kumo = c < ssa and c < ssb
        bearish_ts_below_ks = ts < ks
        bearish_chikou_below_price_past = chikou_span.iloc[-1-kijun_p] < df_close.iloc[-1-kijun_p] if len(chikou_span) > kijun_p and len(df_close) > kijun_p else False
        bearish_future_kumo = ssa < ssb


        # Signal fort
        if bullish_price_above_kumo and bullish_ts_above_ks and bullish_future_kumo: # Ajouté bullish_chikou_above_price_past pour un signal plus fort
            return 1 # Haussier
        elif bearish_price_below_kumo and bearish_ts_below_ks and bearish_future_kumo: # Ajouté bearish_chikou_below_price_past
            return -1 # Baissier
        return 0 # Neutre
    except Exception:
        # st.warning(f"Erreur Ichimoku: {traceback.format_exc()}")
        return 0 # Neutre en cas d'erreur

# --- Récupération des données ---
@st.cache_data(ttl=60*60*3) # Cache les données pendant 3 heures
def fetch_twelve_data(symbol, interval="4h", outputsize=200):
    """Récupère les données de Twelve Data."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "timezone": "UTC" # Important pour la cohérence
    }
    try:
        response = requests.get(TWELVE_DATA_API_URL, params=params)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        data = response.json()

        if data.get("status") == "error":
            st.error(f"Erreur API Twelve Data pour {symbol}: {data.get('message', 'Erreur inconnue')}")
            return None

        if "values" not in data or not data["values"]:
            st.warning(f"Aucune donnée 'values' reçue de Twelve Data pour {symbol}.")
            return None

        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.iloc[::-1] # Inverser pour avoir les dates les plus anciennes en premier
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête pour {symbol}: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du traitement des données pour {symbol}: {e}")
        # st.error(traceback.format_exc()) # Pour un débogage plus détaillé
        return None

# --- Logique du Scanner ---
def scan_pair_td(pair_name):
    """Scanne une paire de devises et retourne les signaux."""
    df = fetch_twelve_data(pair_name, interval="4h", outputsize=300) # Plus de données pour les indicateurs

    if df is None or df.empty:
        st.warning(f"Aucune donnée pour {pair_name}, impossible de scanner.")
        return {
            "Paire": pair_name, "Prix Actuel": np.nan, "Signal HullMA": "N/A",
            "Signal RSI": "N/A", "Signal ADX": "N/A", "Signal Heiken Ashi Lissé": "N/A",
            "Signal Ichimoku": "N/A", "Confluence": "N/A", "Heure Scan (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
        }

    try:
        # S'assurer que les colonnes existent et sont au bon type
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise KeyError(f"Colonne manquante: {col} pour {pair_name}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['open', 'high', 'low', 'close']) # Supprimer les lignes avec NaN dans OHLC
        if df.empty:
            st.warning(f"Données OHLC invalides ou vides après nettoyage pour {pair_name}.")
            raise ValueError("DataFrame vide après nettoyage OHLC.")


        # Calcul des indicateurs
        hma = hull_ma_pine(df['close'], p=20)
        rsi = rsi_pine(df['close'], p=10)
        adx = adx_pine(df['high'], df['low'], df['close'], p=14)
        sho, shc = smoothed_heiken_ashi_pine(df.rename(columns=str.capitalize), l1=10, l2=10) # Nécessite Open, High, Low, Close
        ichimoku_signal = ichimoku_pine_signal(df['high'], df['low'], df['close'])

        # Vérifier si les séries d'indicateurs sont valides et ont une dernière valeur
        last_hma = hma.iloc[-1] if not hma.empty else np.nan
        last_close = df['close'].iloc[-1]
        last_rsi = rsi.iloc[-1] if not rsi.empty else np.nan
        last_adx = adx.iloc[-1] if not adx.empty else np.nan
        last_sho = sho.iloc[-1] if not sho.empty else np.nan
        last_shc = shc.iloc[-1] if not shc.empty else np.nan

        # Détermination des signaux individuels
        signal_hma_val = 0
        if not np.isnan(last_hma) and not np.isnan(last_close):
            if last_close > last_hma: signal_hma_val = 1  # Achat
            elif last_close < last_hma: signal_hma_val = -1 # Vente
        signal_hma_str = "Achat" if signal_hma_val == 1 else ("Vente" if signal_hma_val == -1 else "Neutre")


        signal_rsi_val = 0
        if not np.isnan(last_rsi):
            if last_rsi > 55: signal_rsi_val = 1 # Tendance haussière (peut être surachat si > 70)
            elif last_rsi < 45: signal_rsi_val = -1 # Tendance baissière (peut être survente si < 30)
        signal_rsi_str = "Haussier" if signal_rsi_val == 1 else ("Baissier" if signal_rsi_val == -1 else "Neutre")


        signal_adx_val = 0
        if not np.isnan(last_adx) and last_adx > 20: # ADX > 20 indique une tendance
            signal_adx_val = 1 # Tendance présente (la direction est donnée par d'autres indicateurs)
        signal_adx_str = "Tendance Forte" if signal_adx_val == 1 else "Pas de Tendance / Faible"

        signal_ha_val = 0
        if not np.isnan(last_sho) and not np.isnan(last_shc):
            if last_shc > last_sho: signal_ha_val = 1 # Achat (bougie verte)
            elif last_shc < last_sho: signal_ha_val = -1 # Vente (bougie rouge)
        signal_ha_str = "Achat" if signal_ha_val == 1 else ("Vente" if signal_ha_val == -1 else "Neutre")

        signal_ichimoku_str = "Achat" if ichimoku_signal == 1 else ("Vente" if ichimoku_signal == -1 else "Neutre")

        # Logique de confluence (exemple simple, à affiner selon votre stratégie)
        confluence_score = 0
        if signal_hma_val == 1: confluence_score +=1
        if signal_hma_val == -1: confluence_score -=1
        if signal_rsi_val == 1: confluence_score +=1
        if signal_rsi_val == -1: confluence_score -=1
        if signal_ha_val == 1: confluence_score +=1
        if signal_ha_val == -1: confluence_score -=1
        if ichimoku_signal == 1: confluence_score +=1
        if ichimoku_signal == -1: confluence_score -=1
        # ADX confirme la force, pas la direction directement pour ce score simple
        # if signal_adx_val == 1: confluence_score = confluence_score * 1.2 # Bonus si tendance

        confluence_signal = "N/A"
        # Vous pouvez définir des seuils plus stricts
        if confluence_score >= 3 and signal_adx_val == 1: # Ex: au moins 3 signaux haussiers + tendance
            confluence_signal = "ACHAT FORT CONFLUENCE"
        elif confluence_score <= -3 and signal_adx_val == 1: # Ex: au moins 3 signaux baissiers + tendance
            confluence_signal = "VENTE FORTE CONFLUENCE"
        elif confluence_score >= 2 :
             confluence_signal = "ACHAT CONFLUENCE"
        elif confluence_score <= -2:
            confluence_signal = "VENTE CONFLUENCE"


        return {
            "Paire": pair_name,
            "Prix Actuel": f"{last_close:.5f}",
            "Signal HullMA": signal_hma_str,
            "Signal RSI": signal_rsi_str,
            "Signal ADX": signal_adx_str,
            "Signal Heiken Ashi Lissé": signal_ha_str,
            "Signal Ichimoku": signal_ichimoku_str,
            "Confluence": confluence_signal,
            "Heure Scan (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
        }
    except KeyError as e: # Gérer les colonnes manquantes
        st.error(f"Erreur de clé (colonne manquante probable) pour {pair_name}: {e}")
        return {"Paire": pair_name, "Confluence": "Erreur Données", "Heure Scan (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}
    except ValueError as e: # Gérer les données vides après nettoyage ou indicateurs
        st.error(f"Erreur de valeur (données insuffisantes/invalides) pour {pair_name}: {e}")
        return {"Paire": pair_name, "Confluence": "Erreur Calcul", "Heure Scan (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}
    except Exception as e:
        st.error(f"Erreur inattendue lors du scan de {pair_name}: {e}")
        # st.error(traceback.format_exc()) # Utile pour le débogage
        return {"Paire": pair_name, "Confluence": "Erreur Inconnue", "Heure Scan (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}

# --- Interface Streamlit ---
if 'scan_results_df' not in st.session_state:
    st.session_state.scan_results_df = pd.DataFrame()

if st.button("🔄 Lancer/Actualiser le Scan Forex"):
    if API_KEY: # Vérifier à nouveau au cas où il aurait été invalidé entre-temps
        progress_bar = st.progress(0)
        status_text = st.empty()
        scan_results = []

        for i, pair in enumerate(FOREX_PAIRS_TD):
            status_text.text(f"Scan en cours pour {pair}...")
            result = scan_pair_td(pair)
            scan_results.append(result)
            progress_bar.progress((i + 1) / len(FOREX_PAIRS_TD))
            time.sleep(0.2) # Petite pause pour éviter de surcharger l'API si beaucoup de paires

        status_text.text("Scan terminé !")
        progress_bar.empty()

        if scan_results:
            st.session_state.scan_results_df = pd.DataFrame(scan_results)
        else:
            st.warning("Aucun résultat de scan à afficher.")
            st.session_state.scan_results_df = pd.DataFrame() # Réinitialiser si vide
    else:
        st.error("Impossible de lancer le scan : Clé API non configurée.")

# Afficher la table des résultats (même si vide au début)
if not st.session_state.scan_results_df.empty:
    st.subheader("Résultats du Scan")
    
    # Option pour filtrer les signaux de confluence forts
    show_strong_only = st.checkbox("Afficher uniquement les signaux de confluence FORTS")
    
    df_to_display = st.session_state.scan_results_df
    if show_strong_only:
        df_to_display = df_to_display[df_to_display['Confluence'].str.contains("FORT", na=False)]

    if not df_to_display.empty:
        st.dataframe(df_to_display, use_container_width=True)
    elif show_strong_only:
        st.info("Aucun signal de confluence FORT trouvé avec les filtres actuels.")
    else:
        st.info("Aucun résultat à afficher.") # Devrait être couvert par le check initial
        
else:
    st.info("Cliquez sur 'Lancer/Actualiser le Scan Forex' pour voir les résultats.")

st.markdown("---")
st.markdown(f"*Dernière actualisation des données affichées : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} (heure de l'interface)*")
st.caption("Les signaux fournis sont à titre informatif et ne constituent pas des conseils en investissement.")
