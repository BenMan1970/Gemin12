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
        print(f"Erreur Hull MA: {e}")
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
        print(f"Erreur RSI: {e}")
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
        print(f"Erreur ADX: {e}")
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
        print(f"Erreur Heiken Ashi: {e}")
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
        print(f"Erreur Smoothed Heiken Ashi: {e}")
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if len(df_high) < min_len_req or len(df_low) < min_len_req or len(df_close) < min_len_req:
            print(f"Ichi: Data < ({len(df_close)}) vs req {min_len_req}.")
            return 0
        
        ts = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        ks = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
        sa = (ts + ks) / 2
        sb = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
        
        if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
            print("Ichi: NaN close/spans.")
            return 0
        
        ccl = df_close.iloc[-1]
        cssa = sa.iloc[-1]
        cssb = sb.iloc[-1]
        ctn = max(cssa, cssb)
        cbn = min(cssa, cssb)
        
        if ccl > ctn:
            return 1
        elif ccl < cbn:
            return -1
        else:
            return 0
    except Exception as e:
        print(f"Erreur Ichimoku: {e}")
        return 0

@st.cache_data(ttl=300)
def get_data_twelve(symbol_td: str, interval_td: str = '4h', period_days: int = 15, max_retries: int = 3):
    retry_count = 0
    while retry_count < max_retries:
        print(f"\n--- Début get_data_twelve: sym='{symbol_td}', interval='{interval_td}', period='{period_days}d', tentative {retry_count + 1}/{max_retries} ---")
        try:
            # Vérifier si la clé API existe dans les secrets
            if "TWELVE_DATA_API_KEY" not in st.secrets:
                st.error("❌ Clé API Twelve Data manquante. Ajoutez-la dans les secrets Streamlit.")
                return None
            
            api_key = st.secrets["TWELVE_DATA_API_KEY"]
            
            # Calculate start and end dates
            end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            start_date = (datetime.now(timezone.utc) - timedelta(days=period_days)).strftime('%Y-%m-%d')

            # Twelve Data API endpoint
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol_td,
                'interval': interval_td,
                'start_date': start_date,
                'end_date': end_date,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:  # Rate limit exceeded
                wait_time = 60
                st.warning(f"⏳ Limite de requêtes atteinte pour {symbol_td}. Attente de {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
                continue
            elif response.status_code != 200:
                st.error(f"❌ Erreur API pour {symbol_td}: Code {response.status_code}")
                return None
            
            data = response.json()
            
            # Vérifier si les données sont valides
            if 'status' in data and data['status'] == 'error':
                st.error(f"❌ Erreur API: {data.get('message', 'Erreur inconnue')}")
                return None
            
            if not data.get('values'):
                st.warning(f"⚠️ Pas de données disponibles pour {symbol_td}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df.set_index('datetime', inplace=True)
            
            # Convertir les colonnes en nombres
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.columns = ['Open', 'High', 'Low', 'Close']
            df = df.sort_index()  # Tri chronologique
            
            # Supprimer les lignes avec des valeurs manquantes
            df = df.dropna()

            if len(df) < 50:
                st.warning(f"⚠️ Données insuffisantes pour {symbol_td} ({len(df)} barres)")
                return None

            print(f"✅ Données pour {symbol_td} OK. {len(df)} barres récupérées.")
            return df
            
        except requests.exceptions.Timeout:
            st.error(f"⏰ Timeout pour {symbol_td}")
            retry_count += 1
        except Exception as e:
            st.error(f"❌ Erreur pour {symbol_td}: {str(e)}")
            print(f"Erreur détaillée: {traceback.format_exc()}")
            retry_count += 1
    
    st.error(f"❌ Échec après {max_retries} tentatives pour {symbol_td}")
    return None

def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        print(f"calculate_all_signals: Données insuffisantes ({len(data) if data is not None else 'None'} lignes)")
        return None
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print("calculate_all_signals: Colonnes OHLC manquantes")
        return None
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    
    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}

    # Hull MA
    try:
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev:
                bull_confluences += 1
                signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev:
                bear_confluences += 1
                signal_details_pine['HMA'] = "▼"
            else:
                signal_details_pine['HMA'] = "─"
        else:
            signal_details_pine['HMA'] = "N/A"
    except Exception as e:
        signal_details_pine['HMA'] = "Err"
        print(f"Erreur HMA: {e}")

    # RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >= 1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1
                signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1
                signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else:
                signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else:
            signal_details_pine['RSI'] = "N/A"
            signal_details_pine['RSI_val'] = "N/A"
    except Exception as e:
        signal_details_pine['RSI'] = "Err"
        signal_details_pine['RSI_val'] = "N/A"
        print(f"Erreur RSI: {e}")

    # ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20:
                # ADX confirme la tendance mais ne donne pas de direction
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else:
                signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else:
            signal_details_pine['ADX'] = "N/A"
            signal_details_pine['ADX_val'] = "N/A"
    except Exception as e:
        signal_details_pine['ADX'] = "Err"
        signal_details_pine['ADX_val'] = "N/A"
        print(f"Erreur ADX: {e}")

    # Heiken Ashi
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >= 1 and len(ha_close) >= 1 and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['HA'] = "▼"
            else:
                signal_details_pine['HA'] = "─"
        else:
            signal_details_pine['HA'] = "N/A"
    except Exception as e:
        signal_details_pine['HA'] = "Err"
        print(f"Erreur HA: {e}")

    # Smoothed Heiken Ashi
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >= 1 and len(sha_close) >= 1 and \
           not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['SHA'] = "▼"
            else:
                signal_details_pine['SHA'] = "─"
        else:
            signal_details_pine['SHA'] = "N/A"
    except Exception as e:
        signal_details_pine['SHA'] = "Err"
        print(f"Erreur SHA: {e}")

    # Ichimoku
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1:
            bull_confluences += 1
            signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1:
            bear_confluences += 1
            signal_details_pine['Ichi'] = "▼"
        else:
            signal_details_pine['Ichi'] = "─"
    except Exception as e:
        signal_details_pine['Ichi'] = "Err"
        print(f"Erreur Ichimoku: {e}")
    
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences:
        direction = "HAUSSIER"
    elif bear_confluences > bull_confluences:
        direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"
        
    return {
        'confluence_P': confluence_value, 
        'direction_P': direction, 
        'bull_P': bull_confluences, 
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"), 
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    stars_map = {
        6: "⭐⭐⭐⭐⭐⭐",
        5: "⭐⭐⭐⭐⭐",
        4: "⭐⭐⭐⭐",
        3: "⭐⭐⭐",
        2: "⭐⭐",
        1: "⭐"
    }
    return stars_map.get(confluence_value, "WAIT")

# Interface utilisateur
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox(
        "Confluence minimum (0-6)",
        options=[0, 1, 2, 3, 4, 5, 6],
        index=3,
        format_func=lambda x: f"{x} étoile{'s' if x > 1 else ''}"
    )
    show_all = st.checkbox("Voir toutes les paires (ignorer filtre)")
    pair_to_debug = st.selectbox(
        "🔍 Afficher OHLC pour:",
        ["Aucune"] + FOREX_PAIRS_TD,
        index=0
    )
    scan_btn = st.button(
        "🔍 Scanner (Données Twelve Data H4)",
        type="primary",
        use_container_width=True
    )

with col2:
    if scan_btn:
        st.info("🔄 Scan en cours (Twelve Data H4)...")
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Affichage des données de debug si demandé
        if pair_to_debug != "Aucune":
            st.subheader(f"📊 Données OHLC pour {pair_to_debug} (Twelve Data):")
            debug_data = get_data_twelve(pair_to_debug, interval_td="4h", period_days=5)
            if debug_data is not None:
                st.dataframe(debug_data[['Open', 'High', 'Low', 'Close']].tail(10))
            else:
                st.warning(f"Impossible de charger les données pour {pair_to_debug}")
            st.divider()
        
        # Scan de toutes les paires
        for i, symbol in enumerate(FOREX_PAIRS_TD):
            pair_name = symbol.replace('/', '')
            current_progress = (i + 1) / len(FOREX_PAIRS_TD)
            progress_bar.progress(current_progress)
            status_text.text(f"Analyse: {pair_name} ({i + 1}/{len(FOREX_PAIRS_TD)})")
            
            # Récupération des données
            data_h4 = get_data_twelve(symbol, interval_td="4h", period_days=15)
            
            if data_h4 is not None:
                signals = calculate_all_signals_pine(data_h4)
                if signals:
                    stars = get_stars_pine(signals['confluence_P'])
                    result = {
                        'Paire': pair_name,
                        'Direction': signals['direction_P'],
                        'Conf. (0-6)': signals['confluence_P'],
                        'Étoiles': stars,
                        'RSI': signals['rsi_P'],
                        'ADX': signals['adx_P'],
                        'Bull': signals['bull_P'],
                        'Bear': signals['bear_P'],
                        'details': signals['signals_P']
                    }
                    results.append(result)
                else:
                    results.append({
                        'Paire': pair_name,
                        'Direction': 'ERREUR CALCUL',
                        'Conf. (0-6)': 0,
                        'Étoiles': 'N/A',
                        'RSI': 'N/A',
                        'ADX': 'N/A',
                        'Bull': 0,
                        'Bear': 0,
                        'details': {'Info': 'Calcul des signaux échoué'}
                    })
            else:
                results.append({
                    'Paire': pair_name,
                    'Direction': 'ERREUR DONNÉES',
                    'Conf. (0-6)': 0,
                    'Étoiles': 'N/A',
                    'RSI': 'N/A',
                    'ADX': 'N/A',
                    'Bull': 0,
                    'Bear': 0,
                    'details': {'Info': 'Données Twelve Data indisponibles'}
                })
            
            # Respecter les limites de taux (plan gratuit: 8 req/min)
            if i < len(FOREX_PAIRS_TD) - 1:  # Pas d'attente après la dernière requête
                time.sleep(8)  # 8 secondes entre les requêtes
        
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        if results:
            df_all = pd.DataFrame(results)
            df_filtered = df_all[df_all['Conf. (0-6)'] >= min_conf].copy() if not show_all else df_all.copy()
            
            if not show_all:
                st.success(f"🎯 {len(df_filtered)} paire(s) avec {min_conf}+ confluence")
            else:
                st.info(f"🔍 Affichage des {len(df_filtered)} paires")
            
            if not df_filtered.empty:
                # Tri par confluence décroissante
                df_sorted = df_filtered.sort_values('Conf. (0-6)', ascending=False)
                display_columns = [
                    'Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 
                    'RSI', 'ADX', 'Bull', 'Bear'
                ]
                st.dataframe(
                    df_sorted[display_columns],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Détails des signaux
                with st.expander("📊 Détails des signaux"):
                    for _, row in df_sorted.iterrows():
                        signals_detail = row.get('details', {})
                        if not isinstance(signals_detail, dict):
                            signals_detail = {'Info': 'Détails non disponibles'}
                        
                        st.write(f"**{row.get('Paire', 'N/A')}** - {row.get('Étoiles', 'N/A')} "
                                f"({row.get('Conf. (0-6)', 'N/A')}) - Direction: {row.get('Direction', 'N/A')}")
                        
                        # Affichage des signaux individuels
                        cols = st.columns(6)
                        signal_order = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                        for idx, signal_key in enumerate(signal_order):
                            cols[idx].metric(
                                label=signal_key,
                                value=signals_detail.get(signal_key, "N/A")
                            )
                        st.divider()
            else:
                st.warning("❌ Aucune paire ne correspond aux critères de filtrage")
        else:
            st.error("❌ Aucun résultat obtenu")

# Informations additionnelles
with st.expander("ℹ️ Informations"):
    st.markdown("""
    **Indicateurs utilisés:**
    - **HMA**: Hull Moving Average (tendance)
    - **RSI**: Relative Strength Index (momentum)
    - **ADX**: Average Directional Index (force de tendance)
    - **HA**: Heiken Ashi (tendance lissée)
    - **SHA**: Smoothed Heiken Ashi (tendance très lissée)
    - **Ichi**: Ichimoku (support/résistance)
    
    **Confluence**: Nombre d'indicateurs en accord
    - ⭐⭐⭐⭐⭐⭐ : 6 indicateurs (très fort)
    - ⭐⭐⭐⭐⭐ : 5 indicateurs (fort)
    - ⭐⭐⭐⭐ : 4 indicateurs (moyen)
    - ⭐⭐⭐ : 3 indicateurs (faible)
    """)
                     
