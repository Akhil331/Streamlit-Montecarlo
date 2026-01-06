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

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the dashboard locally:

```bash
streamlit run analysis.py
```

## Strategy Details

The core philosophy is that financial returns are not normally distributed; they exhibit higher kurtosis (more extreme values). By using a Student-t distribution, we generate risk scenarios that are more realistic. Furthermore, by conditioning the *shape* of the daily candle (OHLC) on the *magnitude* of the return, the simulation preserves the market's microstructure behavior (e.g., large down days often have specific volume and wick characteristics compared to small drift days).
