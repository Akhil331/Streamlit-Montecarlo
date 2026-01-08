import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import t, multivariate_t

st.set_page_config(page_title="Monte Carlo Analysis", layout="wide")

st.title("Monte Carlo Simulation with Student-t Distribution")

# Sidebar
st.sidebar.header("Configuration")

# Initialize ticker list in session state if not present
if 'mc_tickers' not in st.session_state:
    st.session_state.mc_tickers = ["^NSEI"]

def add_ticker():
    st.session_state.mc_tickers.append("SPY")

def remove_ticker(index):
    if index < len(st.session_state.mc_tickers):
        st.session_state.mc_tickers.pop(index)

st.sidebar.subheader("Asset Universe")

# Store (Ticker, OHLC_Enabled)
asset_config = []

# Primary Asset (Index 0)
st.sidebar.caption("Primary (Traded)")
primary = st.sidebar.text_input("Ticker Symbol", value=st.session_state.mc_tickers[0], key="input_0", label_visibility="collapsed")
st.session_state.mc_tickers[0] = primary
p_ohlc = st.sidebar.checkbox("Generate OHLC", value=True, key="ohlc_0")
st.sidebar.markdown("---")

asset_config.append((primary, p_ohlc))

# Dynamic Factors (Index 1+)
if len(st.session_state.mc_tickers) > 1:
    st.sidebar.caption("Correlated Factors")

for i in range(1, len(st.session_state.mc_tickers)):
    c1, c2 = st.sidebar.columns([0.85, 0.15], vertical_alignment="bottom")
    with c1:
        val = st.text_input(f"Factor {i}", value=st.session_state.mc_tickers[i], key=f"input_{i}", label_visibility="collapsed", placeholder="Ticker...")
        st.session_state.mc_tickers[i] = val
    with c2:
        if st.button("✕", key=f"del_{i}", help="Remove Factor"):
            remove_ticker(i)
            st.rerun()
    
    # Checkbox below to avoid cluttering horizontal space
    f_ohlc = st.sidebar.checkbox(f"Generate OHLC for Factor {i}", value=False, key=f"ohlc_{i}")
    st.sidebar.markdown("---")
    
    asset_config.append((val, f_ohlc))

if st.sidebar.button("➕ Add Factor"):
    add_ticker()
    st.rerun()

# Guardrails
if len(st.session_state.mc_tickers) > 5:
    st.sidebar.warning("⚠️ High dimensionality (>5) may reduce robustness.")

simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=1000, value=200, step=10)
years = st.sidebar.slider("Time Horizon (Years)", min_value=1, max_value=10, value=1, step=1)
time_horizon = int(years * 252)


@st.cache_data
def get_data(ticker, period="10y"):
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        return data
    except Exception as e:
        return pd.DataFrame()

# Create Tabs
tab1, tab2 = st.tabs(["Simulation Analysis", "How to Use / About"])

with tab1:
    # Unpack config: Filter empty strings, keep (ticker, ohlc) map
    valid_config = [ac for ac in asset_config if ac[0].strip()]
    
    tickers = [t[0].strip().upper() for t in valid_config]
    ohlc_map = {t[0].strip().upper(): t[1] for t in valid_config}
    
    if not tickers:
        st.warning("Please enter at least one valid ticker.")
    else:
        try:
            # 1. Fetch & Align Data
            data_map = {}
            with st.spinner(f"Fetching and aligning data for {len(tickers)} assets..."):
                # Fetch
                valid_tickers = []
                for t_sym in tickers:
                    df = get_data(t_sym)
                    if df.empty or 'Close' not in df.columns:
                        st.error(f"Failed to fetch data for {t_sym}")
                        continue
                    
                    # Cleanup
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    # Calculate Log Returns & Metrics needed later
                    df['PrevClose'] = df['Close'].shift(1)
                    df['LogReturns'] = np.log(df['Close'] / df['PrevClose'])
                    
                    # Pre-calculate OHLC metrics for all (needed if we simulate OHLC)
                    df['DailyRange'] = df['High'] - df['Low']
                    safe_range = df['DailyRange'].replace(0, np.nan)
                    df['BodyAbs'] = (df['Close'] - df['Open']).abs()
                    df['BodyRatio'] = df['BodyAbs'] / safe_range
                    df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
                    df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
                    df['UpperWickRatio'] = df['UpperWick'] / safe_range
                    df['LowerWickRatio'] = df['LowerWick'] / safe_range
                    df['RangePct'] = df['DailyRange'] / df['PrevClose']
                    df['AbsRet'] = df['LogReturns'].abs()
                    df['LogVol'] = np.log(df['Volume'].replace(0, np.nan).fillna(1.0))

                    data_map[t_sym] = df
                    valid_tickers.append(t_sym)

                if not valid_tickers:
                    st.stop()

                # Align Dates (Intersection)
                aligned_index = data_map[valid_tickers[0]].index
                for t_sym in valid_tickers[1:]:
                    aligned_index = aligned_index.intersection(data_map[t_sym].index)
                
                # Check data sufficiency
                if len(aligned_index) < 252:
                    st.error("Insufficient overlapping data (< 1 year).")
                    st.stop()
                
                # Filter data
                for t_sym in valid_tickers:
                    data_map[t_sym] = data_map[t_sym].loc[aligned_index].dropna()

            st.success(f"Market State: {len(valid_tickers)} assets | {len(aligned_index)} days aligned data.")

            # 2. Multivariate Fit
            st.subheader("Multivariate Distribution Analysis")
            
            # Prepare Return Matrix
            return_matrix = pd.DataFrame({t: data_map[t]['LogReturns'] for t in valid_tickers})
            
            # Stats
            mu_vec = return_matrix.mean()
            cov_matrix = return_matrix.cov()
            
            # Correlation Check
            corr_matrix = return_matrix.corr()
            if len(valid_tickers) > 1:
                with st.expander("Correlation Matrix", expanded=True):
                    # Use Plotly Heatmap instead of Matplotlib
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu', 
                        zmin=-1, zmax=1,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text}",
                        showscale=True
                    ))
                    fig_corr.update_layout(
                        title="Correlation Matrix",
                        height=400 + (len(valid_tickers) * 20),
                        width=600
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    max_corr = 0
                    for i in range(len(valid_tickers)):
                        for j in range(i+1, len(valid_tickers)):
                            val = corr_matrix.iloc[i, j]
                            if abs(val) > 0.8:
                                max_corr = max(max_corr, abs(val))
                    
                    if max_corr > 0.8:
                        st.info(f"ℹ️ High correlation detected ({max_corr:.2f}). Factors are strongly coupled.")

            # Degrees of Freedom Estimation
            # CRITICAL FIX: Use min(dofs) instead of mean.
            # Averaging marginal DoFs underestimates tail risk for the joint system.
            # The "weakest link" (heaviest tail) should govern the joint extreme potential.
            dofs = []
            for t_sym in valid_tickers:
                d, _, _ = t.fit(return_matrix[t_sym])
                dofs.append(d)
                
            # Use the most conservative (lowest) DoF to preserve heavy tails
            start_dof = np.min(dofs)
            
            # Display Params
            c1, c2, c3 = st.columns(3)
            c1.metric("System Degrees of Freedom", f"{start_dof:.2f}")
            c2.metric("Primary Volatility (Ann.)", f"{np.sqrt(cov_matrix.iloc[0,0]) * np.sqrt(252):.2%}")
            c3.metric("Primary Mean Return (Ann.)", f"{mu_vec.iloc[0] * 252:.2%}")

            # 3. Simulation Setup
            st.subheader("Monte Carlo Execution")
            
            use_drift = st.checkbox("Override Primary Drift?", value=True)
            
            # Default: Zero drift for all factors to act as pure conditioning variables
            # We only apply drift to the Primary Asset if requested.
            # This prevents factor trends from dominating the simulation over long horizons.
            sim_mu = np.zeros(len(valid_tickers)) 
            
            if use_drift:
                target_ret = st.number_input("Target Annual Return (Primary)", value=0.12, step=0.01)
                daily_drift = np.log(1 + target_ret) / 252
                sim_mu[0] = daily_drift
            
            # 4. Generate Joint Paths
            # Shape Matrix for Multivariate T
            # Cov = shape * df / (df-2)  => shape = Cov * (df-2)/df
            if start_dof > 2:
                shape_matrix = cov_matrix.values * (start_dof - 2) / start_dof
            else:
                shape_matrix = cov_matrix.values # Fallback if df <= 2 (undefined var)

            total_steps = time_horizon * simulations
            
            # Generate Random Shocks (Centered at 0)
            # Size = (TotalSteps, K)
            random_shocks = multivariate_t.rvs(
                loc=np.zeros(len(valid_tickers)), 
                shape=shape_matrix, 
                df=start_dof, 
                size=total_steps
            )
            
            # If 1D (single simulation/asset), reshape ensures 2D
            if len(random_shocks.shape) == 1:
                random_shocks = random_shocks.reshape(-1, 1)

            # 5. Path Reconstruction & OHLC
            st.text("Reconstructing market structure...")
            
            final_simulation_data = [] # List of DataFrames (one per path) ?? No, memory heavy.
            # Better strategy: Store arrays, build DF only on export or sample view.
            
            # Structure to hold results:
            # dict[ticker] -> { 'Close': (time, sims), 'Open': ..., 'High': ... }
            sim_results = {}
            for t_sym in valid_tickers:
                sim_results[t_sym] = {} 

            # Process each asset
            for i, t_sym in enumerate(valid_tickers):
                # Reshape shocks to (Time, Sims)
                # Apply drift
                asset_shocks = random_shocks[:, i].reshape(time_horizon, simulations) + sim_mu[i]
                
                last_price = data_map[t_sym]['Close'].iloc[-1]
                cum_ret = np.cumsum(asset_shocks, axis=0)
                sim_close = last_price * np.exp(cum_ret)
                
                sim_results[t_sym]['Close'] = sim_close
                sim_results[t_sym]['Returns'] = asset_shocks # Log returns
                
                # Conditional OHLC
                # Generate if Enabled in Map (Default False if not found)
                if ohlc_map.get(t_sym, False):
                    # OHLC Logic (from original code, adapted)
                    # 1. Fit Volume Model
                    fit_df = data_map[t_sym][['LogVol', 'AbsRet', 'RangePct', 'BodyRatio', 'UpperWickRatio', 'LowerWickRatio']].dropna()
                    
                    if len(fit_df) > 10:
                        A_mat = np.vstack([fit_df['AbsRet'], np.ones(len(fit_df))]).T
                        try:
                            v_slope, v_int = np.linalg.lstsq(A_mat, fit_df['LogVol'], rcond=None)[0]
                            v_resid = fit_df['LogVol'] - (v_slope * fit_df['AbsRet'] + v_int)
                            v_std = v_resid.std()
                        except:
                            v_slope, v_int, v_std = 0, np.log(100000), 0.5
                    else:
                        v_slope, v_int, v_std = 0, np.log(100000), 0.5

                    sim_abs_ret = np.abs(asset_shocks)

                    # 2. Advanced Binning: Split Downside/Upside Quantiles
                    # We compute quantiles separately for Down and Up days.
                    # This guarantees we have resolution in the tails (Crash vs Rally) 
                    # even if the distribution is skewed.
                    
                    hist_log_ret = data_map[t_sym]['LogReturns'].dropna()
                    neg_rets = hist_log_ret[hist_log_ret < 0]
                    pos_rets = hist_log_ret[hist_log_ret >= 0]
                    
                    try:
                        if len(neg_rets) > 20 and len(pos_rets) > 20:
                            # 5 bins for negative sides (0, 20, 40, 60, 80%)
                            # Note: percentiles are sorted low to high.
                            # neg_q[0] is most negative (extreme). neg_q[-1] is near 0.
                            neg_q = np.percentile(neg_rets, np.linspace(0, 100, 6))
                            
                            # 5 bins for positive side (0, 20, ... 100%)
                            # pos_q[0] is near 0. pos_q[-1] is extreme positive.
                            pos_q = np.percentile(pos_rets, np.linspace(0, 100, 6))
                            
                            # Combine boundaries.
                            # We merge neg_q (excluding last which is ~0) and pos_q (excluding first which is ~0)
                            # To avoid overlap/duplicates at 0, we take mid or simple splice.
                            # Edges: [Min, 20%, 40%, 60%, 80%, 0(approx), 20%+, ...]
                            
                            # Robust merge:
                            # neg_q[:-1] -> [Min, ..., 80%]
                            # pos_q[1:]  -> [20%, ..., Max]
                            # And we insert 0.0 explicitly as the divider? -> pos_q[0] is usually 0.
                            
                            edges = np.concatenate([neg_q[:-1], pos_q])
                            edges = np.sort(np.unique(edges)) # Safety sort
                            
                            # Ensure full coverage
                            edges[0] -= 0.001
                            edges[-1] += 0.001
                            
                        else:
                            # Fallback for sparse data
                            edges = np.percentile(hist_log_ret, np.linspace(0, 100, 11))
                            edges[0] -= 0.001
                            edges[-1] += 0.001
                            
                        # Assign bins to history
                        fit_df['RetBin'] = pd.cut(hist_log_ret, bins=edges, labels=False, include_lowest=True)
                        
                        # Handle simulation mapping
                        # np.digitize returns index i such that bins[i-1] <= x < bins[i]
                        # pd.cut returns 0 for first interval. np.digitize returns 1.
                        # We align them.
                        
                        sim_signed_ret = asset_shocks # Raw Signed Log Return
                        
                        # digitize: bins must be monotonic increasing.
                        sim_bins = np.digitize(sim_signed_ret, edges) - 1
                        
                        # Clamp bins to valid range (0 to len(edges)-2)
                        # Because digitize returns 0 if < edges[0] and len(edges) if > edges[-1]
                        n_bins = len(edges) - 1
                        sim_bins = np.clip(sim_bins, 0, n_bins - 1)

                    except Exception as e:
                        fit_df['RetBin'] = 0
                        sim_bins = np.zeros(sim_signed_ret.shape, dtype=int)
                        # fallback edges? unnecessary here if we fallback to bin 0
                    
                    # Create Candle Library
                    candle_lib = {}
                    for b in fit_df['RetBin'].unique():
                        # fit_df has cols: RangePct, BodyRatio, UpperWickRatio
                        subset = fit_df[fit_df['RetBin'] == b]
                        candle_lib[b] = subset[['RangePct', 'BodyRatio', 'UpperWickRatio']].values
                    
                    # Allocate arrays
                    s_rnge = np.zeros_like(sim_abs_ret)
                    s_body = np.zeros_like(sim_abs_ret)
                    s_uppr = np.zeros_like(sim_abs_ret)
                    
                    unique_b = np.unique(sim_bins)
                    for b_idx in unique_b:
                        mask = (sim_bins == b_idx)
                        count = np.sum(mask)
                        pool = candle_lib.get(b_idx, candle_lib.get(0, np.empty((0,3))))
                        if len(pool) == 0: continue
                        
                        picks = pool[np.random.randint(0, len(pool), size=count)]
                        s_rnge[mask] = picks[:, 0]
                        s_body[mask] = picks[:, 1]
                        s_uppr[mask] = picks[:, 2]

                    # 4. Reconstruction
                    # PrevClose
                    prev_c = np.vstack([np.full((1, simulations), last_price), sim_close[:-1, :]])
                    
                    sim_ranges = prev_c * s_rnge
                    sim_bodies = sim_ranges * s_body
                    sim_upper = sim_ranges * s_uppr
                    
                    signs = np.sign(asset_shocks)
                    signs[signs == 0] = 1
                    
                    sim_open = sim_close - (signs * sim_bodies)
                    max_oc = np.maximum(sim_open, sim_close)
                    sim_high = max_oc + sim_upper
                    sim_low = sim_high - sim_ranges # Strict geometry: H - Range = L
                    
                    # Volume
                    v_noise = np.random.normal(0, v_std, size=(time_horizon, simulations))
                    sim_vol = np.exp(v_slope * sim_abs_ret + v_int + v_noise)

                    sim_results[t_sym]['Open'] = sim_open
                    sim_results[t_sym]['High'] = sim_high
                    sim_results[t_sym]['Low'] = sim_low
                    sim_results[t_sym]['Volume'] = sim_vol

            # 6. Visualization (Primary Asset)
            primary_tk = valid_tickers[0]
            future_dates = pd.date_range(start=data_map[primary_tk].index[-1] + pd.Timedelta(days=1), periods=time_horizon, freq='B')
            
            st.subheader(f"Projected Close: {primary_tk}")
            fig_mc = go.Figure()
            
            p_close = sim_results[primary_tk]['Close']
            mean_path = np.mean(p_close, axis=1)
            
            # Plot 50 paths
            for i in range(min(simulations, 50)):
                fig_mc.add_trace(go.Scatter(x=future_dates, y=p_close[:, i], mode='lines', 
                                            line=dict(width=1, color='rgba(100, 100, 250, 0.2)'), showlegend=False))
            
            fig_mc.add_trace(go.Scatter(x=future_dates, y=mean_path, mode='lines', name='Mean Path', line=dict(color='orange', width=2)))
            st.plotly_chart(fig_mc, use_container_width=True)

            # Sample Candle
            if 'Open' in sim_results[primary_tk]:
                st.subheader("Sample Path Structure")
                idx = 0
                fig_c = go.Figure(go.Candlestick(
                    x=future_dates,
                    open=sim_results[primary_tk]['Open'][:, idx],
                    high=sim_results[primary_tk]['High'][:, idx],
                    low=sim_results[primary_tk]['Low'][:, idx],
                    close=sim_results[primary_tk]['Close'][:, idx],
                    name=f"Sim {idx}"
                ))
                st.plotly_chart(fig_c, use_container_width=True)

            # 7. Export
            st.write("### Export Simulation Data")
            export_count = st.number_input("Paths to Export", 1, simulations, min(10, simulations))
            if st.button("Generate & Export to DB"):
                import os, time
                base_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DB", "monte_carlo")
                ts = time.strftime("%Y%m%d_%H%M%S")
                # Folder: DB/monte_carlo/{PrimaryTicker}/{Timestamp}/
                safe_primary = primary_tk.replace("^", "").replace(".", "_")
                exp_path = os.path.join(base_folder, safe_primary, ts)
                os.makedirs(exp_path, exist_ok=True)
                
                bar = st.progress(0)
                
                for sim_i in range(export_count):
                    # Build Joint DataFrame
                    df_run = pd.DataFrame({'Date': future_dates})
                    
                    for t_sym in valid_tickers:
                        # Naming convention:
                        # Primary Asset -> Standard names (Close, Open, etc.) to preserve strategy compatibility
                        # Factors -> Prefixed names (SPY_Close, SPY_Returns)
                        if t_sym == primary_tk:
                            col_base = ""
                        else:
                            col_base = f"{t_sym}_"
                        
                        # Close & Returns (Always present)
                        df_run[f"{col_base}Close"] = sim_results[t_sym]['Close'][:, sim_i]
                        df_run[f"{col_base}Returns"] = sim_results[t_sym]['Returns'][:, sim_i]
                        
                        # OHLCV (Conditional)
                        if 'Open' in sim_results[t_sym]:
                            df_run[f"{col_base}Open"] = sim_results[t_sym]['Open'][:, sim_i]
                            df_run[f"{col_base}High"] = sim_results[t_sym]['High'][:, sim_i]
                            df_run[f"{col_base}Low"] = sim_results[t_sym]['Low'][:, sim_i]
                            df_run[f"{col_base}Volume"] = sim_results[t_sym]['Volume'][:, sim_i]
                    
                    # Use 'universe' filename but user might expect 'ticker' filename for single asset?
                    # Prompt: "Coherent market universes". "Treats single-asset MC as special case".
                    # Existing structure: safe_ticker_{i}.csv
                    # New structure: safe_primary_{i}.csv (if single) or universe_{i}.csv?
                    # Let's stick to consistent naming. If the strategy reads the folder of the Primary Ticker,
                    # it probably iterates over *.csv files. Naming checks might exist.
                    # Best bet: Keep {safe_primary}_{sim_i+1}.csv
                    
                    f_name = f"{safe_primary}_{sim_i+1}.csv"
                    df_run.to_csv(os.path.join(exp_path, f_name), index=False)
                    bar.progress((sim_i+1)/export_count)
                
                st.success(f"Exported {export_count} simulation files to `{exp_path}`")
                st.write(f"Preview (`{safe_primary}_1.csv`):")
                st.dataframe(pd.read_csv(os.path.join(exp_path, f"{safe_primary}_1.csv")).head())

            # Stats (Primary)
            final_p = p_close[-1, :]
            init_p = data_map[primary_tk]['Close'].iloc[-1]
            exp_r = (final_p.mean() - init_p) / init_p
            
            st.write(f"### {primary_tk} End-State Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Final Price", f"{final_p.mean():.2f}")
            c2.metric("Median Final Price", f"{np.median(final_prices):.2f}" if 'final_prices' in locals() else f"{np.median(final_p):.2f}")
            c3.metric("Expected Return", f"{exp_r:.2%}")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.write("Check your tickers or internet connection.")


with tab2:
    st.markdown("""
    # How to Use / About
    
    This tool is a **strategy truth-filter**. It is designed to test if your trading strategy is robust to alternative market histories (Multiverse Analysis). Unlike simple backtests that run on one path (history), this engine generates thousands of statistically plausible market universes where correlations, trends, and volatilities behave realistically but differently.

    ## Key Features

    ### 1. Multivariate Simulation (New!)
    *   **Joint Probability**: Simulates multiple assets (e.g., NIFTY, VIX, Gold) simultaneously.
    *   **Correlation vs Causation**: The engine preserves the probability of *co-movement* (correlation). Note that this does **not** imply deterministic reaction. A -5% drop in Factor A does not *cause* a drop in B; it simply makes it statistically likely based on history.
    *   **Student-t Fat Tails**: Uses a multivariate Student-t distribution to generate "Black Swan" potential. It captures the fact that extreme events happen more often than a Bell Curve predicts.

    ### 2. Market State & OHLCV
    *   **Primary Asset Geometry**: Generates full Open, High, Low, Close, and Volume data based on Conditional logic (e.g., large down days have specific wick/volume profiles).
    *   **Factor Support**: You can add external factors (like Interest Rates, Global Indices) to condition your strategy.
    *   **OHLC Toggles**: Enable full candle generation for factors only if your strategy explicitly trades or calculates levels on them.

    ### 3. Usage Workflow
    1.  **Define Universe**: Enter your **Primary Asset** (the one you trade) in the sidebar.
    2.  **Add Factors**: Click "➕ Add Factor" to include correlated assets (e.g., `SPY` to correlate with US markets).
    3.  **Configure**: Set the time horizon and number of paths (simulations).
    4.  **Analyze**: View the "Correlation Matrix" and "Projected Paths".
    5.  **Export**: Click "Generate & Export" to save coherent market universes. Each file (e.g., `universe_1.csv`) contains the synchronized prices for ALL assets in that simulation path.

    ## Why usage this?
    Evaluating a strategy on a single historical path is dangerous (Overfitting). By running your strategy across 200+ simulated futures, you can calculate:
    *   **Survivability**: Probability of ruin.
    *   **Robustness**: Does alpha persist if correlations shift slightly?
    
    *Monte Carlo decides truth, not creativity.*
    """)
