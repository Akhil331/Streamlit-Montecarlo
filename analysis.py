import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import t

st.set_page_config(page_title="Monte Carlo Analysis", layout="wide")

st.title("Monte Carlo Simulation with Student-t Distribution")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker", value="^NSEI")
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
    if ticker:
        ticker = ticker.strip()
        try:
            # 1. Fetch Data
            with st.spinner("Fetching data..."):
                data = get_data(ticker)
            
            if data.empty:
                st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
                st.write("Debug info: returned dataframe is empty.")
            else:
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    # The columns are likely (Price, Ticker). We want just Price.
                    data.columns = data.columns.get_level_values(0)
                
                # Ensure 'Close' is present
                if 'Close' not in data.columns:
                     st.error(f"Data found but 'Close' column missing. Columns: {data.columns.tolist()}")
                     st.stop()


                # 2. Plot Price History
                st.subheader(f"Price History: {ticker}")
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig_price.update_layout(title=f"{ticker} Daily Closing Price", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_price, use_container_width=True)

                # 3. Calculate Log Returns
                data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
                log_returns = data['LogReturns'].dropna()

                # 4. Fit Student-t Distribution
                st.subheader("Distribution Analysis")
                
                # Fitting
                # params returns (df, loc, scale)
                params = t.fit(log_returns)
                dof, loc, scale = params
                
                # Display parameters
                col1, col2, col3 = st.columns(3)
                col1.metric("Degrees of Freedom (v)", f"{dof:.4f}")
                col2.metric("Location (μ)", f"{loc:.4f}")
                col3.metric("Scale (σ)", f"{scale:.4f}")

                # 5. Plot Histogram with Fitted Student-t
                # Generate x values for the PDF line
                x_min = log_returns.min()
                x_max = log_returns.max()
                x_range = np.linspace(x_min, x_max, 1000)
                pdf_values = t.pdf(x_range, df=dof, loc=loc, scale=scale)

                fig_dist = go.Figure()
                
                # Histogram
                fig_dist.add_trace(go.Histogram(
                    x=log_returns, 
                    histnorm='probability density', 
                    name='Historical Log Returns',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                # Fitted PDF
                fig_dist.add_trace(go.Scatter(
                    x=x_range, 
                    y=pdf_values, 
                    mode='lines', 
                    name=f'Fitted Student-t (v={dof:.2f})',
                    line=dict(color='red', width=2)
                ))

                fig_dist.update_layout(
                    title="Log Returns Histogram with Student-t Fit",
                    xaxis_title="Log Return",
                    yaxis_title="Density"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # 6. Monte Carlo Simulation
                st.subheader("Monte Carlo Simulation (Advanced Conditional OHLCV)")
                
                # --- Advanced OHLC & Volume Modeling Prep ---
                # 1. Calculate Historical Metrics
                data['PrevClose'] = data['Close'].shift(1)
                data['DailyRange'] = data['High'] - data['Low']
                # Avoid division by zero
                safe_range = data['DailyRange'].replace(0, np.nan) 
                
                data['BodyAbs'] = (data['Close'] - data['Open']).abs()
                data['BodyRatio'] = data['BodyAbs'] / safe_range
                data['UpperWick'] = data['High'] - data[['Open', 'Close']].max(axis=1)
                data['LowerWick'] = data[['Open', 'Close']].min(axis=1) - data['Low']
                data['UpperWickRatio'] = data['UpperWick'] / safe_range
                data['LowerWickRatio'] = data['LowerWick'] / safe_range
                
                # Range Factor: Range / PrevClose (Causal volatility)
                # This is more robust as it relates volatility to the starting price of the day
                data['RangePct'] = data['DailyRange'] / data['PrevClose']
                
                # Volume Modeling
                data['AbsRet'] = data['LogReturns'].abs()
                data['LogVol'] = np.log(data['Volume'].replace(0, np.nan)) # Handle 0 volume
                
                # Drop NaNs for fitting (PrevClose creates NaNs at start)
                fit_df = data[['LogVol', 'AbsRet', 'RangePct', 'BodyRatio', 'UpperWickRatio', 'LowerWickRatio']].dropna()
                
                # Fit Volume Regression
                if len(fit_df) > 10:
                    A = np.vstack([fit_df['AbsRet'], np.ones(len(fit_df))]).T
                    vol_slope, vol_intercept = np.linalg.lstsq(A, fit_df['LogVol'], rcond=None)[0]
                    vol_residuals = fit_df['LogVol'] - (vol_slope * fit_df['AbsRet'] + vol_intercept)
                    vol_std = vol_residuals.std()
                else:
                    vol_slope, vol_intercept, vol_std = 0, np.log(100000), 1.0

                # --- Conditioning Logic (Quantile Binning) ---
                # We must condition candle shape on the magnitude of the return.
                # Large returns -> Large ranges, specific shapes.
                # Small returns -> Small ranges, doji/indecision.
                
                # We create 10 buckets of AbsRet magnitude
                # fit_df['RetBin'] will hold the bin index (0-9)
                try:
                    fit_df['RetBin'] = pd.qcut(fit_df['AbsRet'], q=10, labels=False, duplicates='drop')
                    # Determine the bin edges for simulation mapping
                    # We need to map simulated abs returns to these bins
                    # We can use pd.qcut to get bins, but we need to apply it to new data.
                    # Let's compute quantiles explicitly
                    quantiles = np.percentile(fit_df['AbsRet'], np.linspace(0, 100, 11))
                    quantiles[-1] += 0.0001 # Make inclusive
                except Exception as e:
                    # Fallback if too little data
                    fit_df['RetBin'] = 0
                    quantiles = [0, 1]

                # Store candle library as a dictionary of arrays, keyed by bin index
                candle_library_bins = {}
                unique_bins = fit_df['RetBin'].unique()
                for b in unique_bins:
                    subset = fit_df[fit_df['RetBin'] == b]
                    candle_library_bins[b] = subset[['RangePct', 'BodyRatio', 'UpperWickRatio']].values

                last_price = data['Close'].iloc[-1]
                
                # --- Drift Adjustment ---
                st.markdown("""
                **Drift Adjustment**:
                Override the expected annual return to correct for historical bull-market bias.
                """)
                use_manual_drift = st.checkbox("Override Expected Annual Return (Drift)?", value=True)
                
                if use_manual_drift:
                    target_annual_return = st.number_input("Target Annual Return (Decimal, e.g. 0.12 for 12%)", value=0.12, step=0.01)
                    daily_drift = np.log(1 + target_annual_return) / 252
                    sim_loc = daily_drift
                else:
                    sim_loc = loc

                # Generate Core Price Paths using Student-t
                random_shocks = t.rvs(df=dof, loc=0, scale=scale, size=(time_horizon, simulations))
                random_log_returns = sim_loc + random_shocks
                
                # Cumulative returns for Close prices
                cumulative_log_returns = np.cumsum(random_log_returns, axis=0)
                close_paths = last_price * np.exp(cumulative_log_returns)
                
                # --- Generate OHLCV ---
                st.text("Generating Conditional OHLCV structures...")
                
                # 1. Sample Candle Shapes Conditionally
                # Map simulated returns to bins
                sim_abs_returns = np.abs(random_log_returns)
                
                # Vectorized binning: np.searchsorted finds insertion points
                sim_bins = np.digitize(sim_abs_returns, quantiles[1:-1]) 
                
                # Now we must sample from candle_library_bins[b] for each b.
                # Since vectorizing dictionary lookup with jagged arrays is hard, we iterate.
                
                s_range_pct = np.zeros_like(sim_abs_returns)
                s_body_ratio = np.zeros_like(sim_abs_returns)
                s_upper_ratio = np.zeros_like(sim_abs_returns)
                
                for b_idx in range(len(unique_bins)):
                    # Find all simulation points that fall into bin b_idx
                    # (Note: unique_bins might not include all 0-9 if data is sparse, but usually ok)
                    mask = (sim_bins == b_idx)
                    count = np.sum(mask)
                    
                    if count > 0:
                        # Get empirical pool for this bin
                        pool = candle_library_bins.get(b_idx, candle_library_bins[unique_bins[0]]) # Fallback
                        
                        # Random sampling with replacement
                        if len(pool) > 0:
                            chosen_indices = np.random.randint(0, len(pool), size=count)
                            chosen_metrics = pool[chosen_indices]
                            
                            s_range_pct[mask] = chosen_metrics[:, 0]
                            s_body_ratio[mask] = chosen_metrics[:, 1]
                            s_upper_ratio[mask] = chosen_metrics[:, 2]
                
                # 2. Reconstruct OHLC
                # Previous Close (Lagged Close Paths). 
                prev_close_paths = np.vstack([np.full((1, simulations), last_price), close_paths[:-1, :]])
                
                sim_signs = np.sign(random_log_returns)
                sim_signs[sim_signs == 0] = 1 
                
                # Calculate Range in Price terms
                # Range = PrevClose * RangePct (Correct Causal normalization)
                sim_ranges = prev_close_paths * s_range_pct
                
                # Calculate Body in Price terms
                # Body = Range * BodyRatio
                sim_bodies = sim_ranges * s_body_ratio
                
                # Calculate Open
                # Open = Close - (Sign * Body)
                open_paths = close_paths - (sim_signs * sim_bodies)
                
                # Calculate High / Low
                # UpperWickLen = Range * UpperRatio
                sim_upper_wicks = sim_ranges * s_upper_ratio
                
                # Logic: High = max(Open, Close) + UpperWick
                # Low = High - Range
                max_oc = np.maximum(open_paths, close_paths)
                high_paths = max_oc + sim_upper_wicks
                low_paths = high_paths - sim_ranges
                
                # 3. Simulate Volume
                # log(V) = a + b*|ret| + noise
                vol_noise = np.random.normal(0, vol_std, size=(time_horizon, simulations))
                sim_log_vol = (vol_slope * sim_abs_returns) + vol_intercept + vol_noise
                volume_paths = np.exp(sim_log_vol)
                
                # --- Visualization ---
                # Plot Mean Path (Close)
                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=time_horizon, freq='B')
                
                fig_mc = go.Figure()
                # Plot first 50 paths (Close) lightly
                mean_path = np.mean(close_paths, axis=1)
                
                for i in range(min(simulations, 50)):
                    fig_mc.add_trace(go.Scatter(
                        x=future_dates, y=close_paths[:, i], mode='lines', 
                        line=dict(width=1, color='rgba(100, 100, 250, 0.2)'), showlegend=False
                    ))

                fig_mc.add_trace(go.Scatter(
                    x=future_dates, y=mean_path, mode='lines', name='Mean Close',
                    line=dict(color='orange', width=3)
                ))
                
                fig_mc.update_layout(title=f"Projected Close Prices ({simulations} Paths)", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_mc, use_container_width=True)

                # Sample Candlestick Chart
                st.subheader("Sample Path Visualization (Candlestick)")
                sample_idx = 0 # Visualize the first simulation path
                
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=future_dates,
                    open=open_paths[:, sample_idx],
                    high=high_paths[:, sample_idx],
                    low=low_paths[:, sample_idx],
                    close=close_paths[:, sample_idx],
                    name='Simulated Path 0'
                )])
                fig_candle.update_layout(title="Single Simulated Path Structure", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_candle, use_container_width=True)

                # --- Export ---
                st.write("### Export Simulation Data")
                export_count = st.number_input("Number of Paths to Export", min_value=1, max_value=simulations, value=min(10, simulations))
                
                if st.button("Generate & Export to DB"):
                    import os
                    import time
                    
                    base_db_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DB", "monte_carlo")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # Create a specific folder for this batch, nested by Ticker
                    safe_ticker = ticker.replace("^", "").replace(".", "_")
                    # Structure: DB/monte_carlo/NSEI/20260101_120000/
                    batch_folder = os.path.join(base_db_folder, safe_ticker, timestamp)
                    os.makedirs(batch_folder, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(f"Starting export to {batch_folder}...")
                    
                    for i in range(export_count):
                        # Construct DataFrame for this path
                        df_sim = pd.DataFrame({
                            'Date': future_dates,
                            'Open': open_paths[:, i],
                            'High': high_paths[:, i],
                            'Low': low_paths[:, i],
                            'Close': close_paths[:, i],
                            'Volume': volume_paths[:, i]
                        })
                        
                        # Filename: nsei_1.csv, nsei_2.csv ...
                        filename = f"{safe_ticker}_{i+1}.csv"
                        file_path = os.path.join(batch_folder, filename)
                        
                        df_sim.to_csv(file_path, index=False)
                        progress_bar.progress((i + 1) / export_count)
                    
                    status_text.text("Export complete!")
                    st.success(f"Exported {export_count} simulation files to `{batch_folder}`")
                    
                    if export_count > 0:
                        st.write(f"Preview of `{safe_ticker}_1.csv`:")
                        first_file_path = os.path.join(batch_folder, f"{safe_ticker}_1.csv")
                        st.dataframe(pd.read_csv(first_file_path).head())

                # Final Stats
                final_prices = close_paths[-1, :]
                exp_return = (final_prices.mean() - last_price) / last_price
                
                st.write("### Simulation Statistics (End of Period)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Final Price", f"{final_prices.mean():.2f}")
                m2.metric("Median Final Price", f"{np.median(final_prices):.2f}")
                m3.metric("Expected Return", f"{exp_return:.2%}")


        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab2:
    st.markdown("""
    # Student-t Monte Carlo Simulation for Financial Markets

    This project is a sophisticated Monte Carlo simulation tool designed to model and forecast future stock price movements. Unlike standard simulations that assume a Normal (Gaussian) distribution, this tool uses a **Student-t distribution** to better capture the "fat tails" (extreme events) often observed in financial markets.

    ## Features

    - **Advanced Statistical Modeling**: Fits a Student-t distribution to historical log returns.
    - **Conditional OHLCV Generation**: Not just Close prices! It reconstructs realistic Open, High, Low, and Volume data by modeling the correlation between return magnitude and candle shapes/volume.
    - **Drift Adjustment**: Allows users to override historical drift with a target expected annual return.
    - **Interactive Dashboard**: Built with Streamlit for easy configuration and visualization.
    - **Export Functionality**: Generate and save thousands of simulated trading days (paths) for backtesting strategies.

    ## How it Works

    1.  **Data Ingestion**: Fetches historical data (10 years) using `yfinance`.
    2.  **Distribution Fitting**: Analyzes historical log returns to fit a Student-t distribution (Degrees of Freedom, Location, Scale).
    3.  **Candle Shape Modeling**:
        *   Categorizes days into "bins" based on the magnitude of returns.
        *   Learns the empirical distribution of candle properties (Body size, Wicks, Range) for each bin.
        *   Models Volume as a function of absolute return.
    4.  **Simulation Engine**:
        *   Generates random return paths using the fitted Student-t parameters.
        *   Reconstructs the full OHLCV candle for each simulated day by sampling from the learned "candle library" matching the simulated return.
    5.  **Output**: Visualizes thousands of potential future price paths and allows exporting the detailed data.

    ## Strategy Details

    The core philosophy is that financial returns are not normally distributed; they exhibit higher kurtosis (more extreme values). By using a Student-t distribution, we generate risk scenarios that are more realistic. Furthermore, by conditioning the *shape* of the daily candle (OHLC) on the *magnitude* of the return, the simulation preserves the market's microstructure behavior (e.g., large down days often have specific volume and wick characteristics compared to small drift days).
    """)
