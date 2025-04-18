# TradeProphet

**TradeProphet** is a desktop application built with PySide6 and pyqtgraph for visualizing and analyzing stock and commodity market data, enhanced by AI-driven forecasting and insights. It provides interactive charts, technical indicators, portfolio tracking, and predictive analytics, designed for traders and investors to make informed decisions. The app leverages TensorFlow for machine learning-based forecasts and fetches real-time and historical data using the `yfinance` library, all within a user-friendly, dark-themed interface.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Capabilities](#capabilities)
- [Future Improvements](#future-improvements)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

## Features
- **AI-Powered Forecasting**: Generate price predictions with confidence intervals using TensorFlow-based machine learning models, including metrics like MSE, MAE, and RMSE for accuracy.
- **Interactive Charts**: Visualize price movements, candlestick patterns, and technical indicators with a zoomable and pannable 2D chart.
- **Technical Analysis**: Display Keltner Channels, order blocks, volume delta, and risk/reward zones to identify trading opportunities.
- **Portfolio Management**: Track positions and monitor AI-calculated performance metrics like ROI, Sharpe Ratio, and backtest accuracy.
- **Real-time Data**: Fetch historical and intraday data for stocks, commodities (e.g., US Oil WTI), and other assets using `yfinance`.
- **Customizable UI**: Dark-themed interface with a legend for plot items, built with PySide6 for a modern look and feel.
- **Extensible Design**: Modular architecture supports adding new AI models, indicators, and data sources.

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/tradeprophet.git
   cd tradeprophet
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` should include:
   ```
   PySide6>=6.0.0
   pyqtgraph>=0.13.0
   yfinance>=0.2.0
   pandas>=1.5.0
   numpy>=1.23.0
   tensorflow>=2.10.0
   ta>=0.10.0
   ib_insync>=0.9.0
   ```

4. **Run the Application**:
   ```bash
   python stocks_app_v5.py
   ```

## Usage
1. **Launch the App**:
   Run the script to open the TradeProphet window.

2. **Navigate Tabs**:
   - **Markets Tab**: View a table of assets (e.g., stocks, commodities). Click an asset to display its chart and AI-generated insights.
   - **Portfolio Tab**: Monitor open positions and performance metrics.
   - **Training Tab**: View logs and analysis results, including AI model training details.

3. **Interact with the Chart**:
   - Select an asset and period (e.g., 1 Week, 1 Month) to plot price data and forecasts.
   - Zoom and pan the chart using the mouse.
   - Observe technical indicators like Keltner Channels and order blocks.
   - Use the legend to identify plot items (e.g., Price, Volume Delta).

4. **Analyze Data**:
   - Review AI-driven stats like RSI, Volatility, R/R Ratio, and forecast accuracy in the chart title.
   - Check risk/reward and forecast tables for trading insights based on predictive models.

## Capabilities

TradeProphet combines robust charting with AI-driven analytics to empower traders and investors:

1. **AI-Powered Forecasting and Analysis**:
   - Generate price predictions for 7 to 365 days using TensorFlow-based machine learning models, tailored to the selected period (1 Week, 1 Month, 1 Year, 5 Years).
   - Provide confidence intervals for forecasts, plotted as upper and lower bounds on the chart, to assess prediction reliability.
   - Calculate model performance metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), displayed in the chart title.
   - Evaluate trading strategies with AI-computed metrics, including backtest accuracy, Sharpe Ratio, and Return on Investment (ROI), to guide decision-making.

2. **Interactive 2D Charting**:
   - Plot historical price data as a line graph, with green (uptrend) or red (downtrend) coloring based on trend analysis.
   - Display 15-minute candlestick patterns for 1 Week and 1 Month periods, showing open, high, low, close, and bullish/bearish status.
   - Support zooming and panning to explore data at different scales, with a legend identifying plot items.

3. **Technical Indicators**:
   - **Keltner Channels**: Plot upper, middle, and lower channels to identify volatility and trend direction, currently in gold (to be updated with unique colors).
   - **Order Blocks**: Highlight significant price levels as vertical orange lines, marking institutional buying/selling zones.
   - **Volume Delta**: Show volume changes on a secondary y-axis, plotted in orange.
   - **Risk/Reward Zones**: Display shaded areas for take-profit (green) and stop-loss (red) levels based on user-defined or AI-suggested entries.
   - **Buy/Sell Signals**: Plot scatter points on the chart to indicate AI-generated trading signals based on historical and forecasted data.

4. **User Interface**:
   - Dark-themed UI with a responsive layout, built with PySide6 for a modern experience.
   - Markets tab with a table for selecting assets and periods, triggering chart updates and AI analysis.
   - Chart legend showing plot item names (e.g., Price, Volume Delta, Keltner Upper), added for clarity.
   - Risk/reward and forecast tables summarizing AI-driven trading opportunities and predictions.

5. **Extensibility and Performance**:
   - Threaded data processing with `QThread` to keep the UI responsive during AI computations and data fetching.
   - Modular design for integrating new AI models (e.g., LSTM, ARIMA), indicators (e.g., Bollinger Bands), or data sources (e.g., Alpha Vantage).
   - Support for custom risk/reward inputs via a dialog interface, feeding into AI analysis.

## Future Improvements
- **Enhanced Responsiveness**: Optimize data processing with downsampling and caching, and improve candlestick rendering for large datasets.
- **Interactive Controls**: Add checkboxes to show/hide the legend and chart stats, and a reset button to restore the original zoom.
- **Persistent Stats**: Move chart stats to a `TextItem` on the chart, displayed in two lines and visible across zoom/pan.
- **Distinct Keltner Colors**: Assign unique colors to Keltner Upper, Middle, and Lower channels for clarity.
- **Hover Tooltips**: Implement mouse-over tooltips to display price, date, and volume when hovering over the chart.
- **Crosshair**: Add vertical and horizontal lines to track price and date at the mouse cursor.
- **Data Reliability**: Introduce retries and alternative data sources to handle `yfinance` failures.

## Changelog

### April 18, 2025
- **Added**:
  - Chart legend to the 2D chart in the Markets tab, displaying names of plot items (Price, Volume Delta, Keltner Channels) with a top-left offset.
- **Fixed**:
  - Resolved "QtGui" and "QtCore" not defined errors in CandlestickItem by importing QPicture, QPainter, QPointF, and QRectF directly from PySide6.QtGui and PySide6.QtCore.
- **Attempted (Not Committed)**:
  - Experimented with hover tooltips, crosshair, and title formatting with newline for backtest accuracy, but rolled back due to chart plotting issues.
  - Added debug logging to troubleshoot plotting failures, identifying potential `yfinance` data issues.
- **Notes**:
  - Rolled back to a stable state with the legend to ensure chart functionality.
  - Hover tooltips and crosshair planned for future implementation with improved stability.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests and update documentation for new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
