import sys
import yfinance as yf
import pandas as pd
import json
import os
import hashlib
from datetime import datetime, timedelta
import logging
import numpy as np
import tensorflow as tf
Sequential = tf.keras.Sequential
load_model = tf.keras.models.load_model
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle
from ta.momentum import RSIIndicator
from PySide6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QHBoxLayout, QWidget, QHeaderView, QLineEdit,
                               QComboBox, QToolBar, QLabel, QMessageBox, QDialog, QPushButton,
                               QTabWidget, QMenuBar, QMenu, QFormLayout, QDoubleSpinBox,
                               QProgressBar, QTextEdit)
from PySide6.QtGui import QColor, QFont, QPainter, QWindow, QVector3D
from PySide6.QtCore import Qt, QTimer, QDateTime, QThread, Signal, QObject, Slot
from PySide6.QtDataVisualization import (Q3DSurface, QSurface3DSeries, QSurfaceDataProxy,
                                        QSurfaceDataItem, QHeightMapSurfaceDataProxy,
                                        Q3DTheme, Q3DCamera)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis
from ib_insync import IB, Stock, Forex, Future, Crypto, MarketOrder, LimitOrder
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress yfinance DEBUG logs
logging.getLogger('yfinance').setLevel(logging.WARNING)

# Search Ticker Dialog
class SearchTickerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Search Ticker")
        self.parent = parent
        self.layout = QVBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter name (e.g., Bitcoin, Steel, Coal)")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_tickers)
        self.layout.addWidget(QLabel("Search by Name:"))
        self.layout.addWidget(self.search_input)
        self.layout.addWidget(self.search_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Symbol", "Name", "Type", "Action"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.layout.addWidget(self.results_table)

        self.setLayout(self.layout)

    def search_tickers(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a search term.")
            return

        logger.debug(f"Searching for {query}")
        try:
            ticker = yf.Ticker(query)
            info = ticker.info
            results = [{
                "symbol": info.get("symbol", query),
                "name": info.get("longName", query),
                "type": self.get_asset_type(info)
            }]
            self.display_results(results)
        except Exception as e:
            logger.error(f"Search failed for {query}: {e}")
            QMessageBox.warning(self, "Error", f"Search failed: {e}")

    def get_asset_type(self, info):
        quote_type = info.get("quoteType", "").upper()
        if quote_type == "CRYPTOCURRENCY":
            return "crypto"
        elif quote_type in ["EQUITY", "ETF"]:
            return "stocks"
        elif quote_type == "CURRENCY":
            return "forex"
        elif quote_type == "FUTURE" or info.get("sector") == "Commodities":
            return "commodity"
        return "unknown"

    def display_results(self, results):
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(result["symbol"]))
            self.results_table.setItem(row, 1, QTableWidgetItem(result["name"]))
            self.results_table.setItem(row, 2, QTableWidgetItem(result["type"]))
            add_button = QPushButton("Add")
            add_button.clicked.connect(lambda checked, r=row: self.add_ticker(row))
            self.results_table.setCellWidget(row, 3, add_button)

    def add_ticker(self, row):
        symbol = self.results_table.item(row, 0).text()
        name = self.results_table.item(row, 1).text()
        asset_type = self.results_table.item(row, 2).text()

        if asset_type == "crypto" and any(item["symbol"] == symbol for item in self.parent.cryptos):
            QMessageBox.warning(self, "Error", f"{symbol} already in Cryptos.")
            return
        elif asset_type == "stocks" and any(item["symbol"] == symbol for item in self.parent.stocks):
            QMessageBox.warning(self, "Error", f"{symbol} already in Stocks.")
            return
        elif asset_type in ["commodity", "forex"] and any(item["symbol"] == symbol for item in self.parent.commodities_forex):
            QMessageBox.warning(self, "Error", f"{symbol} already in Commodities & Forex.")
            return

        new_item = {"symbol": symbol, "name": name}
        if asset_type == "crypto":
            self.parent.cryptos.append(new_item)
        elif asset_type == "stocks":
            self.parent.stocks.append(new_item)
            self.parent.filtered_stocks.append(new_item)
        elif asset_type in ["commodity", "forex"]:
            new_item["yfinance"] = symbol
            new_item["finnhub"] = symbol
            self.parent.commodities_forex.append(new_item)

        self.parent.save_symbols()
        self.parent.trigger_update()
        QMessageBox.information(self, "Success", f"Added {symbol} to {asset_type.capitalize()}.")
        self.accept()

# Training Worker
class TrainingWorker(QObject):
    log_message = Signal(str)
    training_finished = Signal()
    progress_updated = Signal(int)  # Signal for training progress

    def __init__(self, parent, symbol, history, seq_length=10, forecast_horizon=3):
        super().__init__()
        self.parent = parent
        self.symbol = symbol
        self.history = history
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

    @Slot()
    def run(self):
        self.log_message.emit(f"Starting training for {self.symbol}...")
        
        try:
            # Use closing prices
            prices = self.history["Close"].values.reshape(-1, 1)

            # Normalize the data
            scaler_file = f"scaler_{self.symbol}.pkl"
            if os.path.exists(scaler_file):
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
                prices_scaled = scaler.transform(prices)
            else:
                scaler = MinMaxScaler()
                prices_scaled = scaler.fit_transform(prices)
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)

            self.log_message.emit(f"Data normalized for {self.symbol} (shape: {prices_scaled.shape})")

            # Create sequences
            X, y = self.create_sequences(prices_scaled)
            self.log_message.emit(f"Created {len(X)} sequences for training")

            # Check if sequences are sufficient
            if len(X) < 2:  # Need at least 2 sequences for train/validation split
                raise ValueError(f"Not enough sequences to train model. Created {len(X)} sequences, need at least 2.")

            # Split into training and validation sets
            train_size = int(len(X) * 0.8)
            if train_size == 0:
                train_size = 1  # Ensure at least one sample for training
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # Reshape for LSTM [samples, timesteps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
            y_val = y_val.reshape((y_val.shape[0], y_val.shape[1]))

            # Define or load the LSTM model
            model_file = f"lstm_{self.symbol}.h5"
            if os.path.exists(model_file):
                try:
                    model = load_model(model_file, compile=False)  # Load without compiling to avoid deserialization issues
                    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())  # Recompile with the loss as an object
                    self.log_message.emit(f"Loaded existing model for {self.symbol}")
                except Exception as e:
                    self.log_message.emit(f"Failed to load model for {self.symbol}: {str(e)}. Creating a new model.")
                    model = Sequential([
                        Input(shape=(self.seq_length, 1)),
                        LSTM(20, return_sequences=True),
                        Dropout(0.2),
                        LSTM(20),
                        Dropout(0.2),
                        Dense(self.forecast_horizon)
                    ])
                    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())  # Use the loss object
                    self.log_message.emit(f"Created new LSTM model for {self.symbol}")
            else:
                model = Sequential([
                    Input(shape=(self.seq_length, 1)),
                    LSTM(20, return_sequences=True),
                    Dropout(0.2),
                    LSTM(20),
                    Dropout(0.2),
                    Dense(self.forecast_horizon)
                ])
                model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())  # Use the loss object
                self.log_message.emit(f"Created new LSTM model for {self.symbol}")

            # Train the model with progress updates
            epochs = 1
            for epoch in range(epochs):
                model.fit(
                    X_train, y_train,
                    epochs=1,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    verbose=0,
                    callbacks=[self.LoggingCallback(self.log_message)]
                )
                # Update progress (epoch-based)
                progress = int(((epoch + 1) / epochs) * 100)
                self.progress_updated.emit(progress)
                QApplication.processEvents()  # Keep UI responsive

            # Save the model
            model.save(model_file)
            self.log_message.emit(f"Model saved for {self.symbol}")

            # Log final metrics
            history = model.evaluate(X_train, y_train, verbose=0), model.evaluate(X_val, y_val, verbose=0)
            train_loss = history[0]
            val_loss = history[1]
            self.log_message.emit(f"Training complete for {self.symbol}. Final train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        except Exception as e:
            self.log_message.emit(f"Error during training for {self.symbol}: {str(e)}")
            import traceback
            self.log_message.emit(traceback.format_exc())
        finally:
            self.training_finished.emit()

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length - self.forecast_horizon + 1):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length:i + self.seq_length + self.forecast_horizon])
        return np.array(X), np.array(y)

    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_signal):
            super().__init__()
            self.log_signal = log_signal

        def on_epoch_end(self, epoch, logs=None):
            self.log_signal.emit(f"Epoch {epoch + 1}: train loss = {logs['loss']:.4f}, val loss = {logs['val_loss']:.4f}")
            
# Training Tab
class TrainingTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()

        self.train_button = QPushButton("Train Models")
        self.train_button.clicked.connect(self.train_all_models)
        self.layout.addWidget(self.train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle %p%")
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                max-width: 300px;
                margin: auto;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)

        self.setLayout(self.layout)

        # Training thread setup
        self.training_thread = QThread()
        self.current_worker = None
        self.symbols_to_train = []
        self.histories = []
        self.current_symbol_index = 0
        self.total_symbols = 0
        self.phase = "fetching"  # Can be "fetching" or "training"

    def log_message(self, message):
        self.log_area.append(message)
        QApplication.processEvents()  # Force UI update

    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.progress_bar.setFormat(f"{message} %p%")
        QApplication.processEvents()  # Force UI update

    def start_training(self, symbol, history):
        if self.training_thread.isRunning():
            self.log_message("Training already in progress. Please wait.")
            return

        self.log_message(f"Starting training thread for {symbol}")
        self.current_worker = TrainingWorker(self.parent, symbol, history)
        self.current_worker.moveToThread(self.training_thread)
        self.current_worker.log_message.connect(self.log_message)
        self.current_worker.progress_updated.connect(self.on_training_progress)
        self.current_worker.training_finished.connect(self.on_training_finished)
        self.training_thread.started.connect(self.current_worker.run)
        self.training_thread.start()
        self.log_message("Training thread started")

        # Add a timeout
        timeout_timer = QTimer()
        timeout_timer.setSingleShot(True)
        timeout_timer.timeout.connect(lambda: self.handle_training_timeout(symbol))
        timeout_timer.start(300000)  # 5 minutes in milliseconds

    def handle_training_timeout(self, symbol):
        if self.training_thread.isRunning():
            self.log_message(f"Training for {symbol} timed out after 5 minutes")
            self.training_thread.quit()
            self.training_thread.wait()
            self.on_training_finished()

    def on_training_progress(self, percentage):
        # Update progress bar during training phase
        if self.phase == "training":
            overall_progress = int((self.current_symbol_index / self.total_symbols) * 100)
            training_progress = (percentage / self.total_symbols)  # Contribution of current symbol
            total_progress = min(100, overall_progress + training_progress)
            self.update_progress(total_progress, f"Training {self.symbols_to_train[self.current_symbol_index]}")

    def on_training_finished(self):
        self.log_message("Training thread finished")
        self.training_thread.quit()
        self.training_thread.wait()

        # Move to the next symbol
        self.current_symbol_index += 1
        if self.current_symbol_index < len(self.symbols_to_train):
            # Update progress for the training phase
            self.phase = "training"
            overall_progress = int((self.current_symbol_index / self.total_symbols) * 100)
            self.update_progress(overall_progress, f"Training {self.symbols_to_train[self.current_symbol_index]}")
            self.start_training(self.symbols_to_train[self.current_symbol_index], self.histories[self.current_symbol_index])
        else:
            # Training complete
            self.update_progress(100, "Training Complete")
            self.phase = "fetching"
            self.current_symbol_index = 0
            self.symbols_to_train = []
            self.histories = []
            self.progress_bar.setVisible(False)
            self.train_button.setEnabled(True)

    def fetch_history_for_symbol(self, symbol):
        # Fetch 5 years of data for training
        history = self.parent.fetch_history(symbol, "5 Years")
        QApplication.processEvents()  # Keep UI responsive
        return history

    def train_all_models(self):
        if self.training_thread.isRunning():
            self.log_message("Training already in progress. Please wait.")
            return

        self.log_area.clear()
        self.log_message("Starting training for all models...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.train_button.setEnabled(False)

        # Collect all symbols
        self.symbols_to_train = []
        for item in self.parent.stocks:
            self.symbols_to_train.append(item["symbol"])
        for item in self.parent.indices:
            self.symbols_to_train.append(item["symbol"])
        for item in self.parent.cryptos:
            self.symbols_to_train.append(item["symbol"])
        for item in self.parent.commodities_forex:
            symbol = item.get("yfinance", item["symbol"])
            self.symbols_to_train.append(symbol)

        # Remove invalid symbol "COAL"
        self.symbols_to_train = [s for s in self.symbols_to_train if s != "COAL"]

        self.total_symbols = len(self.symbols_to_train)
        self.current_symbol_index = 0
        self.histories = []
        self.phase = "fetching"

        # Fetch historical data for all symbols with progress updates
        for idx, symbol in enumerate(self.symbols_to_train):
            self.log_message(f"Fetching historical data for {symbol}...")
            history = self.fetch_history_for_symbol(symbol)
            if history is not None:
                self.histories.append(history)
            else:
                self.log_message(f"Skipping {symbol}: No historical data available.")
                self.symbols_to_train.pop(idx)  # Remove the symbol from the list
                self.total_symbols -= 1  # Adjust total symbols

        if not self.symbols_to_train:
            self.log_message("No valid symbols to train. Aborting.")
            self.progress_bar.setVisible(False)
            self.train_button.setEnabled(True)
            return

        # Start training the first symbol
        self.phase = "training"
        self.current_symbol_index = 0
        self.update_progress(0, f"Training {self.symbols_to_train[0]}")
        self.start_training(self.symbols_to_train[0], self.histories[0])

# Data Update Worker
class DataUpdateWorker(QObject):
    data_updated = Signal(dict)
    chart_updated = Signal(dict)
    progress_updated = Signal(int, str)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.total_steps = 0
        self.current_step = 0

    def update_progress(self, increment, message):
        self.current_step += increment
        if self.total_steps > 0:
            percentage = int((self.current_step / self.total_steps) * 100)
            percentage = min(percentage, 100)
            self.progress_updated.emit(percentage, message)

    @Slot()
    def run(self):
        logger.debug("DataUpdateWorker running")
        try:
            table_data = {}
            total_items = (len(self.parent.indices) + len(self.parent.filtered_stocks) +
                           len(self.parent.cryptos) + len(self.parent.commodities_forex))
            self.total_steps = total_items + 5
            self.current_step = 0

            for table, data, category in [
                (self.parent.indices_table, self.parent.indices, "indices"),
                (self.parent.stocks_table, self.parent.filtered_stocks, "stocks"),
                (self.parent.crypto_table, self.parent.cryptos, "crypto"),
                (self.parent.commodities_forex_table, self.parent.commodities_forex, "commodities_forex")
            ]:
                rows = []
                for item in data:
                    symbol = item["symbol"]
                    name = item["name"]
                    if category == "commodities_forex" and "yfinance" in item:
                        symbol = item["yfinance"]
                    quote = self.parent.fetch_quote(symbol, self)
                    if quote:
                        price = quote["price"]
                        change = ((price - quote["previous_close"]) / quote["previous_close"]) * 100 if quote["previous_close"] else 0
                        rows.append({
                            "symbol": item["symbol"],
                            "name": name,
                            "price": f"{price:.2f}",
                            "change": f"{change:+.2f}%",
                            "change_color": QColor(0, 255, 0) if change >= 0 else Qt.red
                        })
                    else:
                        rows.append({
                            "symbol": item["symbol"],
                            "name": name,
                            "price": "N/A",
                            "change": "N/A",
                            "change_color": Qt.black
                        })
                    self.update_progress(1, "Fetching data")
                table_data[category] = rows

            logger.debug(f"Emitting data_updated with {len(table_data)} categories")
            self.data_updated.emit(table_data)

            if self.parent.current_row is not None and self.parent.current_category is not None:
                items = {"indices": self.parent.indices, "stocks": self.parent.filtered_stocks,
                         "crypto": self.parent.cryptos, "commodities_forex": self.parent.commodities_forex}
                item = items[self.parent.current_category][self.parent.current_row]
                symbol = item["symbol"]
                if self.parent.current_category == "commodities_forex" and "yfinance" in item:
                    symbol = item["yfinance"]
                name = item["name"]
                period = self.parent.period_combo.currentText()

                cache_key = f"{symbol}:{period}"
                now = datetime.now()
                if cache_key in self.parent.last_chart_data and (now - self.parent.last_chart_data[cache_key]["timestamp"]).total_seconds() < 60:
                    logger.debug(f"Using cached chart data for {symbol}:{period}")
                    history = self.parent.last_chart_data[cache_key]["data"].copy()
                    self.update_progress(1, "Fetching data")
                else:
                    history = self.parent.fetch_history(symbol, period, self)
                    if history is None:
                        logger.warning(f"No history for {symbol}")
                        self.chart_updated.emit({"name": name, "error": True})
                        return
                    self.parent.last_chart_data[cache_key] = {"data": history, "timestamp": now}

                if period == "1 Week":
                    latest_quote = self.parent.fetch_quote(symbol, self)
                    if latest_quote:
                        latest_price = latest_quote["price"]
                        latest_time = datetime.now()
                        history.loc[latest_time] = latest_price
                    self.update_progress(1, "Fetching data")

                series_data = [(QDateTime.fromString(str(date), Qt.ISODate).toMSecsSinceEpoch(), price)
                               for date, price in history["Close"].items()]
                self.update_progress(1, "Fetching data")

                prediction_basis = self.parent.prediction_basis_combo.currentText()
                prediction_basis_map = {
                    "1 Week": "7d",
                    "2 Weeks": "14d",
                    "3 Weeks": "21d",
                    "1 Month": "1mo",
                    "3 Months": "3mo"
                }
                prediction_history = self.parent.fetch_history(symbol, prediction_basis, self)
                if prediction_history is None:
                    logger.warning(f"No prediction history for {symbol} with period {prediction_basis}")
                    prediction_history = history
                self.update_progress(1, "Fetching data")

                assessment_history = history
                if self.parent.assessment_period_combo.currentText() == "1 Year":
                    assessment_history = self.parent.fetch_history(symbol, "1 Year", self)
                    if assessment_history is None:
                        logger.warning(f"No 1-year history for {symbol}, using chart period data")
                        assessment_history = history
                    self.update_progress(1, "Fetching data")

                forecast_horizon = self.parent.forecast_horizon_combo.currentText()
                forecast_days_map = {
                    "1 Week": 7,
                    "2 Weeks": 14,
                    "3 Weeks": 21,
                    "1 Month": 30
                }
                forecast_days = forecast_days_map.get(forecast_horizon, 7)

                # Retrain the model with the latest data
                self.parent.training_tab.start_training(symbol, prediction_history)

                # Predict using the (possibly retrained) model
                self.update_progress(0, f"Predicting trend for next {forecast_days} days...")
                forecast = self.parent.predict_with_lstm(symbol, prediction_history, forecast_days)
                rsi = self.parent.calculate_rsi(assessment_history)
                volatility = self.parent.calculate_volatility(assessment_history)
                self.update_progress(1, f"Predicting trend for next {forecast_days} days...")

                forecast_series_data = []
                if forecast is not None:
                    last_date = history.index[-1]
                    last_timestamp = QDateTime.fromString(str(last_date), Qt.ISODate).toMSecsSinceEpoch()
                    for i, price in enumerate(forecast):
                        next_date = last_date + timedelta(days=i + 1)
                        timestamp = QDateTime.fromString(str(next_date), Qt.ISODate).toMSecsSinceEpoch()
                        forecast_series_data.append((timestamp, price))

                growth_potential = "Low"
                risk_level = "Low"
                if forecast is not None:
                    last_price = assessment_history["Close"].iloc[-1]
                    avg_forecast = np.mean(forecast)
                    if avg_forecast > last_price * 1.05:
                        growth_potential = "High"
                    elif avg_forecast > last_price:
                        growth_potential = "Moderate"
                
                if volatility > 0.3:
                    risk_level = "High"
                elif volatility > 0.15:
                    risk_level = "Moderate"
                
                if rsi < 30:
                    growth_potential = "High" if growth_potential != "High" else growth_potential
                elif rsi > 70:
                    risk_level = "High" if risk_level != "High" else risk_level

                analysis = f"Growth: {growth_potential} | Risk: {risk_level} | RSI: {rsi:.2f} | Volatility: {volatility:.2f}"

                logger.debug(f"Emitting chart_updated for {symbol}")
                self.chart_updated.emit({
                    "name": name,
                    "period": period,
                    "series_data": series_data,
                    "forecast_series_data": forecast_series_data,
                    "analysis": analysis,
                    "error": False
                })
        except Exception as e:
            logger.error(f"Worker failed: {e}")

# Trade Dialog
class TradeDialog(QDialog):
    def __init__(self, parent, category, row):
        super().__init__(parent)
        self.setWindowTitle("Place Trade")
        self.layout = QVBoxLayout()
        self.parent = parent
        self.category = category
        self.row = row

        items = {"indices": self.parent.indices, "stocks": self.parent.filtered_stocks,
                 "crypto": self.parent.cryptos, "commodities_forex": self.parent.commodities_forex}
        item = items[category][row]
        symbol = item["symbol"]
        name = item["name"]

        self.layout.addWidget(QLabel(f"Trading {name} ({symbol})"))

        self.action_combo = QComboBox()
        self.action_combo.addItems(["Buy", "Sell"])
        self.layout.addWidget(QLabel("Action:"))
        self.layout.addWidget(self.action_combo)

        self.quantity_input = QLineEdit()
        self.quantity_input.setPlaceholderText("Enter quantity")
        self.layout.addWidget(QLabel("Quantity:"))
        self.layout.addWidget(self.quantity_input)

        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit"])
        self.order_type_combo.currentTextChanged.connect(self.toggle_limit_price)
        self.layout.addWidget(QLabel("Order Type:"))
        self.layout.addWidget(self.order_type_combo)

        self.limit_price_input = QLineEdit()
        self.limit_price_input.setPlaceholderText("Enter limit price")
        self.limit_price_input.setVisible(False)
        self.layout.addWidget(QLabel("Limit Price:"))
        self.layout.addWidget(self.limit_price_input)

        submit_button = QPushButton("Place Order")
        submit_button.clicked.connect(self.submit_order)
        self.layout.addWidget(submit_button)

        self.setLayout(self.layout)

    def toggle_limit_price(self, order_type):
        self.limit_price_input.setVisible(order_type == "Limit")

    def submit_order(self):
        if not self.parent.current_user or not self.parent.ib:
            QMessageBox.warning(self, "Error", "Please login to trade.")
            return

        action = self.action_combo.currentText().lower()
        quantity = self.quantity_input.text()
        order_type = self.order_type_combo.currentText()
        limit_price = self.limit_price_input.text() if order_type == "Limit" else None

        if not quantity.isdigit() or int(quantity) <= 0:
            QMessageBox.warning(self, "Error", "Quantity must be a positive integer.")
            return

        quantity = int(quantity)
        if order_type == "Limit" and (not limit_price or float(limit_price) <= 0):
            QMessageBox.warning(self, "Error", "Invalid limit price.")
            return

        items = {"indices": self.parent.indices, "stocks": self.parent.filtered_stocks,
                 "crypto": self.parent.cryptos, "commodities_forex": self.parent.commodities_forex}
        symbol = items[self.category][self.row]["symbol"]

        if self.category in ["stocks", "indices"]:
            contract = Stock(symbol, 'SMART', 'USD')
        elif self.category == "crypto":
            crypto_symbol = symbol.split("-")[0]
            contract = Crypto(crypto_symbol, 'PAXOS', 'USD')
        elif self.category == "commodities_forex":
            if "USD/EUR" in symbol:
                contract = Forex('EURUSD')
            elif "GOLD" in symbol:
                contract = Future('GC', '202512', 'COMEX')
            elif "SILVER" in symbol:
                contract = Future('SI', '202512', 'COMEX')
            else:
                QMessageBox.warning(self, "Error", "Unsupported asset.")
                return

        self.parent.ib.qualifyContracts(contract)
        order = MarketOrder(action, quantity) if order_type == "Market" else LimitOrder(action, quantity, float(limit_price))

        try:
            trade = self.parent.ib.placeOrder(contract, order)
            QMessageBox.information(self, "Success", "Order placed successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to place order: {e}")

# Withdrawal Dialog
class WithdrawDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Withdraw Funds")
        self.layout = QVBoxLayout()
        self.parent = parent

        disclaimer = QLabel(
            "Withdrawals are processed via Interactive Brokers. A 2% app fee will be applied, separate from IBKR fees."
        )
        disclaimer.setStyleSheet("color: #FF0000;")
        self.layout.addWidget(disclaimer)

        self.amount_input = QDoubleSpinBox()
        self.amount_input.setRange(0, 1000000)
        self.amount_input.setPrefix("$")
        self.layout.addWidget(QLabel("Amount to Withdraw:"))
        self.layout.addWidget(self.amount_input)

        submit_button = QPushButton("Confirm Withdrawal")
        submit_button.clicked.connect(self.process_withdrawal)
        self.layout.addWidget(submit_button)

        self.setLayout(self.layout)

    def process_withdrawal(self):
        amount = self.amount_input.value()
        if amount <= 0:
            QMessageBox.warning(self, "Error", "Please enter a positive amount.")
            return

        app_fee = amount * 0.02
        total_withdrawal = amount - app_fee
        QMessageBox.information(self, "Success", f"Withdrawal processed.\nAmount: ${amount:.2f}\nApp Fee: ${app_fee:.2f}\nTotal Received: ${total_withdrawal:.2f} (excluding IBKR fees)")
        self.accept()

# Authentication Dialog
class AuthDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Authentication")
        self.layout = QVBoxLayout()
        self.parent = parent

        self.tabs = QTabWidget()
        self.register_tab = QWidget()
        self.login_tab = QWidget()
        self.tabs.addTab(self.register_tab, "Register")
        self.tabs.addTab(self.login_tab, "Login")
        self.layout.addWidget(self.tabs)

        register_layout = QVBoxLayout()
        disclaimer = QLabel(
            "Regulatory Disclosure:\n"
            "1. Trading involves risk. 60% of retail investors lose money.\n"
            "2. Funds held by Interactive Brokers, SIPC protected up to $500,000 ($250,000 cash). "
            "Commodities, forex, and crypto not covered.\n"
            "3. App charges a 2% withdrawal fee, separate from IBKR fees."
        )
        disclaimer.setStyleSheet("color: #FF0000;")
        register_layout.addWidget(disclaimer)

        self.register_username = QLineEdit()
        self.register_password = QLineEdit()
        self.register_password.setEchoMode(QLineEdit.Password)
        self.register_ibkr_key = QLineEdit()
        register_layout.addWidget(QLabel("Username:"))
        register_layout.addWidget(self.register_username)
        register_layout.addWidget(QLabel("Password:"))
        register_layout.addWidget(self.register_password)
        register_layout.addWidget(QLabel("IBKR API Key:"))
        register_layout.addWidget(self.register_ibkr_key)

        register_button = QPushButton("Register")
        register_button.clicked.connect(self.register)
        register_layout.addWidget(register_button)
        self.register_tab.setLayout(register_layout)

        login_layout = QVBoxLayout()
        self.login_username = QLineEdit()
        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(QLabel("Username:"))
        login_layout.addWidget(self.login_username)
        login_layout.addWidget(QLabel("Password:"))
        login_layout.addWidget(self.login_password)

        login_button = QPushButton("Login")
        login_button.clicked.connect(self.login)
        login_layout.addWidget(login_button)
        self.login_tab.setLayout(login_layout)

        self.setLayout(self.layout)

    def register(self):
        username = self.register_username.text()
        password = self.register_password.text()
        ibkr_key = self.register_ibkr_key.text()
        if not all([username, password, ibkr_key]):
            QMessageBox.warning(self, "Error", "All fields are required.")
            return

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        user_data = {"username": username, "password": hashed_password, "ibkr_key": ibkr_key}
        with open("users.json", "a") as f:
            json.dump(user_data, f)
            f.write("\n")
        QMessageBox.information(self, "Success", "Registered successfully. Please login.")
        self.tabs.setCurrentIndex(1)

    def login(self):
        username = self.login_username.text()
        password = self.login_password.text()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        if not os.path.exists("users.json"):
            QMessageBox.warning(self, "Error", "No users registered.")
            return

        with open("users.json", "r") as f:
            for line in f:
                user_data = json.loads(line.strip())
                if user_data["username"] == username and user_data["password"] == hashed_password:
                    self.parent.current_user = username
                    self.parent.ib = IB()
                    try:
                        self.parent.ib.connect('127.0.0.1', 7497, clientId=1)
                        QMessageBox.information(self, "Success", "Logged in successfully.")
                        self.parent.update_portfolio()
                        self.accept()
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"IBKR connection failed: {e}")
                    return
        QMessageBox.warning(self, "Error", "Invalid credentials.")

# Portfolio Tab
class PortfolioTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()

        self.portfolio_table = QTableWidget()
        self.portfolio_table.setColumnCount(5)
        self.portfolio_table.setHorizontalHeaderLabels(["Symbol", "Name", "Quantity", "Average Cost", "Market Value"])
        self.portfolio_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.portfolio_table)

        self.balance_label = QLabel("Account Balance: Login to view")
        self.layout.addWidget(self.balance_label)

        self.withdraw_button = QPushButton("Withdraw Funds")
        self.withdraw_button.clicked.connect(self.open_withdraw_dialog)
        self.withdraw_button.setEnabled(False)
        self.layout.addWidget(self.withdraw_button)

        self.setLayout(self.layout)

    def update_portfolio(self):
        if not self.parent.ib:
            self.portfolio_table.setRowCount(0)
            self.balance_label.setText("Account Balance: Login to view")
            return

        try:
            positions = self.parent.ib.positions()
            self.portfolio_table.setRowCount(len(positions))
            for row, position in enumerate(positions):
                contract = position.contract
                ticker = yf.Ticker(contract.symbol)
                name = ticker.info.get("longName", contract.symbol)
                self.portfolio_table.setItem(row, 0, QTableWidgetItem(contract.symbol))
                self.portfolio_table.setItem(row, 1, QTableWidgetItem(name))
                self.portfolio_table.setItem(row, 2, QTableWidgetItem(str(position.position)))
                self.portfolio_table.setItem(row, 3, QTableWidgetItem(f"{position.avgCost:.2f}"))
                self.portfolio_table.setItem(row, 4, QTableWidgetItem(f"{position.marketValue:.2f}"))

            account = self.parent.ib.accountSummary()
            for item in account:
                if item.tag == "TotalCashValue":
                    self.balance_label.setText(f"Account Balance: ${float(item.value):.2f}")
                    break
        except Exception as e:
            logger.error(f"Portfolio update failed: {e}")
            QMessageBox.warning(self, "Error", f"Failed to update portfolio: {e}")

    def open_withdraw_dialog(self):
        if not self.parent.current_user:
            QMessageBox.warning(self, "Error", "Please login to withdraw.")
            return
        dialog = WithdrawDialog(self.parent)
        dialog.exec_()
        self.update_portfolio()


class StockMarketApp(QMainWindow):
    update_triggered = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Market Trends")
        self.setGeometry(100, 100, 1400, 800)

        logger.debug("Initializing StockMarketApp")
        self.api_keys = {"Alpha Vantage": "YOUR_ALPHA_VANTAGE_KEY", "Finnhub": "YOUR_FINNHUB_KEY", "yfinance": None}
        self.cache_file = "stock_cache.json"
        self.symbols_file = "symbols.json"
        self.historical_data_file = "historical_data.json"
        self.cache = self.load_cache()
        self.historical_data = self.load_historical_data()
        self.last_update_time = datetime.min

        self.stocks = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            {"symbol": "TSLA", "name": "Tesla, Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "WMT", "name": "Walmart Inc."},
            {"symbol": "META", "name": "Meta Platforms, Inc."},
            {"symbol": "NFLX", "name": "Netflix, Inc."}
        ]
        self.filtered_stocks = self.stocks.copy()

        self.indices = [
            {"symbol": "SPY", "name": "S&P 500 ETF"},
            {"symbol": "DIA", "name": "Dow Jones ETF"},
            {"symbol": "QQQ", "name": "Nasdaq-100 ETF"},
            {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
            {"symbol": "^NDX", "name": "Nasdaq-100 Index"},
            {"symbol": "^RUT", "name": "Russell 2000 Index"}
        ]

        self.cryptos = [
            {"symbol": "BTC-USD", "name": "Bitcoin"},
            {"symbol": "ETH-USD", "name": "Ethereum"},
            {"symbol": "BNB-USD", "name": "Binance Coin"},
            {"symbol": "XRP-USD", "name": "Ripple"},
            {"symbol": "ADA-USD", "name": "Cardano"}
        ]

        self.commodities_forex = [
            {"symbol": "UKOIL", "name": "UK Oil Brent", "yfinance": "BZ=F", "finnhub": "BZ=F"},
            {"symbol": "USOIL", "name": "US Oil WTI", "yfinance": "CL=F", "finnhub": "CL=F"},
            {"symbol": "SAUDI", "name": "Franklin FTSE Saudi", "yfinance": "FLSA", "finnhub": "FLSA"},
            {"symbol": "BNO", "name": "United States Brent", "yfinance": "BNO", "finnhub": "BNO"},
            {"symbol": "USD/EUR", "name": "USD/EUR", "yfinance": "EURUSD=X", "finnhub": "OANDA:EUR_USD"},
            {"symbol": "GOLD", "name": "Gold", "yfinance": "GC=F", "finnhub": "GC=F"},
            {"symbol": "SILVER", "name": "Silver", "yfinance": "SI=F", "finnhub": "SI=F"}
        ]

        self.load_symbols()

        self.current_user = None
        self.ib = None
        self.current_row = None
        self.current_category = None
        self.last_chart_data = {}
        self.last_valid_data = {}

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        file_menu = QMenu("File", self)
        self.menu_bar.addMenu(file_menu)
        file_menu.addAction("Login", self.open_login_dialog)
        file_menu.addAction("Register", self.open_register_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        self.main_content = QWidget()
        self.main_layout = QVBoxLayout(self.main_content)
        self.tab_widget.addTab(self.main_content, "Markets")

        self.portfolio_tab = PortfolioTab(self)
        self.tab_widget.addTab(self.portfolio_tab, "Portfolio")

        self.training_tab = TrainingTab(self)
        self.tab_widget.addTab(self.training_tab, "Training")

        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)

        self.search_ticker_button = QPushButton("Search Ticker")
        self.search_ticker_button.clicked.connect(self.open_search_ticker_dialog)
        self.toolbar.addWidget(QLabel("Add Asset: "))
        self.toolbar.addWidget(self.search_ticker_button)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Alpha Vantage", "Finnhub", "yfinance"])
        self.provider_combo.setCurrentText("yfinance")
        self.toolbar.addWidget(QLabel("Data Provider: "))
        self.toolbar.addWidget(self.provider_combo)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search Stocks...")
        self.search_bar.textChanged.connect(self.filter_stocks)
        self.toolbar.addWidget(QLabel("Search Stocks: "))
        self.toolbar.addWidget(self.search_bar)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years"])
        self.period_combo.currentTextChanged.connect(self.on_period_changed)
        self.toolbar.addWidget(QLabel("Chart Period: "))
        self.toolbar.addWidget(self.period_combo)

        self.assessment_period_combo = QComboBox()
        self.assessment_period_combo.addItems(["Chart Period", "1 Year"])
        self.assessment_period_combo.setCurrentText("1 Year")
        self.assessment_period_combo.currentTextChanged.connect(self.on_period_changed)
        self.toolbar.addWidget(QLabel("Assessment Period: "))
        self.toolbar.addWidget(self.assessment_period_combo)

        self.forecast_horizon_combo = QComboBox()
        self.forecast_horizon_combo.addItems(["1 Week", "2 Weeks", "3 Weeks", "1 Month"])
        self.forecast_horizon_combo.setCurrentText("1 Week")
        self.forecast_horizon_combo.currentTextChanged.connect(self.on_forecast_horizon_changed)
        self.toolbar.addWidget(QLabel("Forecast Horizon: "))
        self.toolbar.addWidget(self.forecast_horizon_combo)

        self.prediction_basis_combo = QComboBox()
        self.prediction_basis_combo.addItems(["1 Week", "2 Weeks", "3 Weeks", "1 Month", "3 Months"])
        self.prediction_basis_combo.setCurrentText("1 Month")
        self.prediction_basis_combo.currentTextChanged.connect(self.on_prediction_basis_changed)
        self.toolbar.addWidget(QLabel("Prediction Basis: "))
        self.toolbar.addWidget(self.prediction_basis_combo)

        self.content_layout = QHBoxLayout()
        self.tables_layout = QVBoxLayout()

        self.indices_table = QTableWidget()
        self.stocks_table = QTableWidget()
        self.crypto_table = QTableWidget()
        self.commodities_forex_table = QTableWidget()

        category_map = {
            "Indices (Markets)": ("indices", self.indices_table),
            "Stocks": ("stocks", self.stocks_table),
            "Crypto": ("crypto", self.crypto_table),
            "Commodities & Forex": ("commodities_forex", self.commodities_forex_table)
        }

        for category, data, headers in [
            ("Indices (Markets)", self.indices, ["Symbol", "Name", "Value", "Change (%)", "Action"]),
            ("Stocks", self.filtered_stocks, ["Symbol", "Name", "Price ($)", "Change (%)", "Action"]),
            ("Crypto", self.cryptos, ["Symbol", "Name", "Price ($)", "Change (%)", "Action"]),
            ("Commodities & Forex", self.commodities_forex, ["Symbol", "Name", "Value", "Change (%)", "Action"])
        ]:
            header = QLabel(category)
            header.setFont(QFont("Arial", 12, QFont.Bold))
            self.tables_layout.addWidget(header)
            if category in category_map:
                table_category, table = category_map[category]
                table.setRowCount(len(data))
                table.setColumnCount(len(headers))
                table.setHorizontalHeaderLabels(headers)
                table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                table.setEditTriggers(QTableWidget.NoEditTriggers)
                table.cellClicked.connect(lambda r, c, t=table, cat=table_category: self.update_chart_from_table(r, c, t, cat))
                self.tables_layout.addWidget(table)
            else:
                logger.warning(f"Unknown category {category}")

        # Set up the 2D and 3D charts
        self.chart_2d = QChart()
        self.chart_2d_view = QChartView(self.chart_2d)
        self.chart_2d_view.setRenderHint(QPainter.Antialiasing)
        self.chart_2d_view.setMinimumSize(400, 200)

       # self.chart_3d = Q3DSurface()
        #self.chart_3d_container = QWidget.createWindowContainer(self.chart_3d, self)
        #self.chart_3d_container.setMinimumSize(400, 300)

        # Configure the 3D chart
        #self.chart_3d.axisX().setTitle("Time")
        #self.chart_3d.axisY().setTitle("Volume")
        #self.chart_3d.axisZ().setTitle("Price ($)")
        #self.chart_3d.setAspectRatio(2.0)
        #self.chart_3d.setHorizontalAspectRatio(1.0)
        #self.chart_3d.setShadowQuality(Q3DSurface.ShadowQualityLow)
        #self.chart_3d.activeTheme().setType(Q3DTheme.ThemeQt)
        #self.chart_3d.activeTheme().setLabelBackgroundEnabled(False)

        # Add both charts to the layout
        self.chart_widget = QWidget()
        self.chart_container_layout = QVBoxLayout(self.chart_widget)
        self.chart_container_layout.addWidget(self.chart_2d_view)  # 2D chart on top
        #self.chart_container_layout.addWidget(self.chart_3d_container)  # 3D chart below

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Fetching data %p%")
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                max-width: 300px;
                margin: auto;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.progress_bar.setVisible(False)
        self.chart_container_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        self.content_layout.addLayout(self.tables_layout)
        self.content_layout.addWidget(self.chart_widget)
        self.main_layout.addLayout(self.content_layout)

        self.thread = QThread()
        self.worker = DataUpdateWorker(self)
        self.worker.moveToThread(self.thread)
        self.update_triggered.connect(self.worker.run)
        self.thread.started.connect(lambda: logger.debug("Thread started"))
        self.thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.trigger_update)
        self.timer.start(60000)

        self.worker.data_updated.connect(self.apply_table_updates)
        self.worker.chart_updated.connect(self.apply_chart_updates)
        self.worker.progress_updated.connect(self.update_progress)

        self.initialize_tables()
        self.trigger_update()

    def initialize_tables(self):
        logger.debug("Initializing tables")
        for table, data, category in [
            (self.indices_table, self.indices, "indices"),
            (self.stocks_table, self.filtered_stocks, "stocks"),
            (self.crypto_table, self.cryptos, "crypto"),
            (self.commodities_forex_table, self.commodities_forex, "commodities_forex")
        ]:
            table.setRowCount(len(data))
            for row, item in enumerate(data):
                table.setItem(row, 0, QTableWidgetItem(item["symbol"]))
                table.setItem(row, 1, QTableWidgetItem(item["name"]))
                table.setItem(row, 2, QTableWidgetItem("N/A"))
                table.setItem(row, 3, QTableWidgetItem("N/A"))
                trade_button = QPushButton("Trade")
                trade_button.setStyleSheet("background-color: #219fdf; color: white;")
                trade_button.clicked.connect(lambda checked, r=row, c=category: self.open_trade_dialog(c, r))
                table.setCellWidget(row, 4, trade_button)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        return {'date': datetime.now().strftime('%Y-%m-%d'), 'quotes': {}}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load_historical_data(self):
        if os.path.exists(self.historical_data_file):
            try:
                with open(self.historical_data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
        return {}

    def save_historical_data(self):
        try:
            with open(self.historical_data_file, 'w') as f:
                json.dump(self.historical_data, f)
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")

    def load_symbols(self):
        if os.path.exists(self.symbols_file):
            try:
                with open(self.symbols_file, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded symbols: {data}")
                    if hasattr(self, 'stocks'):
                        self.stocks.extend(data.get("stocks", []))
                        self.filtered_stocks = self.stocks.copy()
                    if hasattr(self, 'cryptos'):
                        self.cryptos.extend(data.get("cryptos", []))
                    if hasattr(self, 'commodities_forex'):
                        self.commodities_forex.extend(data.get("commodities_forex", []))
                    if hasattr(self, 'indices'):
                        self.indices.extend(data.get("indices", []))
            except Exception as e:
                logger.error(f"Failed to load symbols: {e}")

    def save_symbols(self):
        data = {
            "stocks": [item for item in self.stocks if item not in self.stocks[:10]],
            "cryptos": [item for item in self.cryptos if item not in self.cryptos[:5]],
            "commodities_forex": [item for item in self.commodities_forex if item not in self.commodities_forex[:7]],
            "indices": [item for item in self.indices if item not in self.indices[:6]]
        }
        try:
            with open(self.symbols_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save symbols: {e}")

    def fetch_quote(self, symbol, worker=None):
        provider = self.provider_combo.currentText()
        cache_key = f"{provider}:{symbol}"
        if cache_key in self.cache['quotes']:
            cache_time = datetime.strptime(self.cache['date'], '%Y-%m-%d')
            if (datetime.now() - cache_time).total_seconds() < 600:
                logger.debug(f"Using cached quote for {symbol}")
                if worker:
                    worker.update_progress(0.5, "Fetching data")
                return self.cache['quotes'][cache_key]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                if worker:
                    worker.update_progress(0.2, "Fetching data")
                for interval in ["1m", "1d"]:
                    quote = ticker.history(period="2d", interval=interval, timeout=30)
                    if not quote.empty:
                        break
                    if worker:
                        worker.update_progress(0.3, "Fetching data")
                else:
                    logger.error(f"{symbol}: No price data found after attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return self.last_valid_data.get(symbol)
                    time.sleep(2)
                    continue
                if worker:
                    worker.update_progress(0.5, "Fetching data")
                latest_price = float(quote["Close"].iloc[-1])
                previous_close = float(quote["Close"].iloc[0] if len(quote) > 1 else latest_price)
                data = {"price": latest_price, "previous_close": previous_close}
                self.cache['quotes'][cache_key] = data
                self.cache['date'] = datetime.now().strftime('%Y-%m-%d')
                self.last_valid_data[symbol] = data
                self.save_cache()
                logger.debug(f"Fetched quote for {symbol}: {data}")
                return data
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return self.last_valid_data.get(symbol)
                time.sleep(2)

    def fetch_history(self, symbol, period, worker=None):
        period_map = {
            "1 Week": ("7d", "1h"),
            "2 Weeks": ("14d", "1h"),
            "3 Weeks": ("21d", "1h"),
            "1 Month": ("1mo", "1d"),
            "3 Months": ("3mo", "1d"),
            "6 Months": ("6mo", "1d"),
            "1 Year": ("1y", "1d"),
            "2 Years": ("2y", "1d"),
            "5 Years": ("5y", "1d")
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                if worker:
                    worker.update_progress(0.2, "Fetching data")
                period_val, interval = period_map[period]
                history = ticker.history(period=period_val, interval=interval, timeout=30)
                if worker:
                    worker.update_progress(0.6, "Fetching data")
                if history.empty:
                    logger.error(f"{symbol}: No history data found for {period} after attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2)
                    continue

                # Append new data to historical dataset
                if symbol not in self.historical_data:
                    self.historical_data[symbol] = []
                existing_dates = {entry["date"] for entry in self.historical_data[symbol]}
                for date, row in history.iterrows():
                    date_str = date.strftime('%Y-%m-%d %H:%M:%S')
                    if date_str not in existing_dates:
                        self.historical_data[symbol].append({
                            "date": date_str,
                            "Close": float(row["Close"]),
                            "Volume": float(row["Volume"])  # Include volume for 3D chart
                        })
                self.save_historical_data()

                logger.debug(f"Fetched history for {symbol}: {len(history)} rows")
                if worker:
                    worker.update_progress(0.2, "Fetching data")
                return history[["Close", "Volume"]]  # Return both Close and Volume
            except Exception as e:
                logger.error(f"Error fetching history for {symbol} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)

    def calculate_rsi(self, history):
        close_prices = history["Close"]
        rsi_indicator = RSIIndicator(close=close_prices, window=14)
        rsi = rsi_indicator.rsi()
        return rsi.iloc[-1]

    def calculate_volatility(self, history):
        returns = history["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        return volatility

    def predict_with_lstm(self, symbol, history, forecast_days, seq_length=10):
        try:
            # Check if model exists
            model_file = f"lstm_{symbol}.h5"
            scaler_file = f"scaler_{symbol}.pkl"
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.warning(f"No trained model for {symbol}. Please train the model first.")
                return None

            # Load the model and scaler
            model = load_model(model_file)
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

            # Prepare the input data
            prices = history["Close"].values.reshape(-1, 1)
            prices_scaled = scaler.transform(prices)

            # Take the last seq_length days as input
            if len(prices_scaled) < seq_length:
                logger.error(f"Not enough data for {symbol}. Required: {seq_length}, Available: {len(prices_scaled)}")
                return None

            input_data = prices_scaled[-seq_length:]
            input_data = input_data.reshape((1, seq_length, 1))

            # Predict
            forecast_scaled = model.predict(input_data, verbose=0)
            forecast_scaled = forecast_scaled.reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast_scaled).flatten()

            # Ensure the forecast length matches forecast_days
            if len(forecast) < forecast_days:
                forecast = np.pad(forecast, (0, forecast_days - len(forecast)), mode='edge')
            elif len(forecast) > forecast_days:
                forecast = forecast[:forecast_days]

            return forecast
        except Exception as e:
            logger.error(f"LSTM prediction failed for {symbol}: {e}")
            return None

    def apply_table_updates(self, table_data):
        logger.debug(f"Applying table updates: {list(table_data.keys())}")
        for table, data, category in [
            (self.indices_table, self.indices, "indices"),
            (self.stocks_table, self.filtered_stocks, "stocks"),
            (self.crypto_table, self.cryptos, "crypto"),
            (self.commodities_forex_table, self.commodities_forex, "commodities_forex")
        ]:
            table.setRowCount(len(data))
            rows = table_data.get(category, [])
            if not rows:
                logger.warning(f"No data for {category}, using defaults")
                rows = [{"symbol": item["symbol"], "name": item["name"], "price": "N/A", "change": "N/A", "change_color": Qt.black}
                        for item in data]
            for row, item in enumerate(rows):
                table.setItem(row, 0, QTableWidgetItem(item["symbol"]))
                table.setItem(row, 1, QTableWidgetItem(item["name"]))
                table.setItem(row, 2, QTableWidgetItem(item["price"]))
                table.setItem(row, 3, QTableWidgetItem(item["change"]))
                table.item(row, 3).setForeground(item["change_color"])
                trade_button = QPushButton("Trade")
                trade_button.setStyleSheet("background-color: #219fdf; color: white;")
                trade_button.clicked.connect(lambda checked, r=row, c=category: self.open_trade_dialog(c, r))
                table.setCellWidget(row, 4, trade_button)
        self.update_portfolio()

    def show_progress_bar(self):
        # Clear the 2D chart
        self.chart_2d.removeAllSeries()
        self.chart_2d.setTitle("Fetching data...")

       

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Fetching data %p%")
        self.progress_bar.setVisible(True)

    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.progress_bar.setFormat(f"{message} %p%")

    def apply_chart_updates(self, chart_data):
        logger.debug(f"Applying chart updates for {chart_data.get('name', 'unknown')}")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

        if chart_data.get("error"):
            self.chart_2d.setTitle(f"{chart_data['name']} (Data Unavailable)")
            self.chart_2d.removeAllSeries()

            
        # --- 2D Chart (Price Trend and Forecast) ---
        # Clear the chart and remove all axes to prevent duplication
        self.chart_2d.removeAllSeries()
        for axis in self.chart_2d.axes():
            self.chart_2d.removeAxis(axis)

        # Create series for historical prices
        price_series = QLineSeries()
        price_series.setName("Price")
        price_series.setColor(Qt.blue)

        min_price = float('inf')
        max_price = float('-inf')
        min_time = float('inf')
        max_time = float('-inf')

        for timestamp, price in chart_data["series_data"]:
            price_series.append(timestamp, price)
            min_price = min(min_price, price)
            max_price = max(max_price, price)
            min_time = min(min_time, timestamp)
            max_time = max(max_time, timestamp)

        self.chart_2d.addSeries(price_series)

        # Create series for forecast
        forecast_series = QLineSeries()
        forecast_series.setName("Forecast")
        forecast_series.setColor(Qt.red)

        if chart_data["forecast_series_data"]:
            for timestamp, price in chart_data["forecast_series_data"]:
                forecast_series.append(timestamp, price)
                min_price = min(min_price, price)
                max_price = max(max_price, price)
                max_time = max(max_time, timestamp)

            self.chart_2d.addSeries(forecast_series)
        else:
            logger.warning("No forecast data available to plot.")

        # Set up axes for 2D chart
        axis_x = QDateTimeAxis()
        axis_x.setFormat("yyyy-MM-dd")
        axis_x.setTitleText("Date")
        axis_x.setMin(QDateTime.fromMSecsSinceEpoch(min_time))
        axis_x.setMax(QDateTime.fromMSecsSinceEpoch(max_time))
        self.chart_2d.addAxis(axis_x, Qt.AlignBottom)
        price_series.attachAxis(axis_x)
        forecast_series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Price ($)")
        axis_y.setLabelFormat("%.2f")
        axis_y.setMin(min_price * 0.95)
        axis_y.setMax(max_price * 1.05)
        self.chart_2d.addAxis(axis_y, Qt.AlignLeft)
        price_series.attachAxis(axis_y)
        forecast_series.attachAxis(axis_y)

        # Set chart title
        self.chart_2d.setTitle(f"{chart_data['name']} Price Trend")

        # --- 3D Chart (Surface with Volume) ---
        # Clear existing series
        while self.chart_3d.seriesList():
            self.chart_3d.removeSeries(self.chart_3d.seriesList()[0])

        # Re-fetch history to get volume data
        items = {"indices": self.indices, "stocks": self.filtered_stocks,
                 "crypto": self.cryptos, "commodities_forex": self.commodities_forex}
        item = items[self.current_category][self.current_row]
        symbol = item["symbol"]
        if self.current_category == "commodities_forex" and "yfinance" in item:
            symbol = item["yfinance"]
        period = self.period_combo.currentText()
        full_history = self.fetch_history(symbol, period)
        if full_history is None:
            logger.warning(f"No history data for {symbol} to plot 3D chart")
            self.chart_3d.axisX().setTitle(f"{symbol} (No Data)")
            self.chart_3d_container.update()
            return

        # Prepare data for 3D surface chart
        history = pd.DataFrame([(ts, price) for ts, price in chart_data["series_data"]], columns=["Timestamp", "Close"])
        history["Timestamp"] = pd.to_datetime(history["Timestamp"], unit='ms')
        history.set_index("Timestamp", inplace=True)

        # Align full_history with chart_data's timestamps
        full_history.index = full_history.index.tz_localize(None)
        history.index = history.index.tz_localize(None)
        full_history = full_history[full_history.index.isin(history.index)]

        # Create surface data
        data_array = []
        x_count = len(full_history)

        volume_scaler = MinMaxScaler(feature_range=(0, 100))
        volumes = full_history["Volume"].values.reshape(-1, 1)
        normalized_volumes = volume_scaler.fit_transform(volumes).flatten()

        for i in range(x_count):
            row = []
            price = float(full_history["Close"].iloc[i])
            volume = float(normalized_volumes[i])
            # Use QVector3D to set (x, y, z) where x is time, y is volume, z is price
            data_item = QSurfaceDataItem(QVector3D(i, volume, price))
            row.append(data_item)
            data_array.append(row)

        # Create surface series for historical data
        proxy = QSurfaceDataProxy()
        proxy.resetArray(data_array)
        series = QSurface3DSeries(proxy)
        series.setDrawMode(QSurface3DSeries.DrawSurfaceAndWireframe)
        series.setFlatShadingEnabled(True)
        series.setBaseColor(Qt.blue)
        self.chart_3d.addSeries(series)

        # Set axis ranges for historical data
        self.chart_3d.axisX().setRange(0, x_count - 1)
        self.chart_3d.axisY().setRange(0, 100)
        self.chart_3d.axisZ().setRange(min(full_history["Close"]), max(full_history["Close"]))

        # Add forecast data as a separate series
        if chart_data["forecast_series_data"]:
            forecast_array = []
            for i, (ts, price) in enumerate(chart_data["forecast_series_data"]):
                row = []
                # Use the last volume value for forecast (since we don't have future volume)
                volume = normalized_volumes[-1]
                data_item = QSurfaceDataItem(QVector3D(x_count + i, volume, price))
                row.append(data_item)
                forecast_array.append(row)

            forecast_proxy = QSurfaceDataProxy()
            forecast_proxy.resetArray(forecast_array)
            forecast_series = QSurface3DSeries(forecast_proxy)
            forecast_series.setDrawMode(QSurface3DSeries.DrawSurfaceAndWireframe)
            forecast_series.setFlatShadingEnabled(True)
            forecast_series.setBaseColor(Qt.red)
            self.chart_3d.addSeries(forecast_series)

            # Update x-axis range to include forecast
            self.chart_3d.axisX().setRange(0, x_count + len(chart_data["forecast_series_data"]) - 1)

        # Set chart title and analysis
        assessment_basis = "1-year data" if self.assessment_period_combo.currentText() == "1 Year" else f"{chart_data['period']} data"
        self.chart_3d.axisX().setTitle(f"Time ({chart_data['period']})")
        self.chart_3d.axisZ().setTitle(f"Price ($) - {chart_data['name']}\n{chart_data['analysis']} (Based on {assessment_basis})")

        self.chart_3d_container.update()

    def filter_stocks(self):
        search_text = self.search_bar.text().lower()
        self.filtered_stocks = [stock for stock in self.stocks if search_text in stock["symbol"].lower() or search_text in stock["name"].lower()]
        self.trigger_update()

    def update_chart_from_table(self, row, column, table, category):
        self.indices_table.clearSelection()
        self.stocks_table.clearSelection()
        self.crypto_table.clearSelection()
        self.commodities_forex_table.clearSelection()
        table.selectRow(row)
        self.current_row = row
        self.current_category = category
        self.last_chart_data = {}
        self.show_progress_bar()
        self.trigger_update()

    def on_period_changed(self):
        if self.current_row is not None and self.current_category is not None:
            self.last_chart_data = {}
            self.show_progress_bar()
            self.trigger_update()

    def on_forecast_horizon_changed(self):
        if self.current_row is not None and self.current_category is not None:
            self.last_chart_data = {}
            self.show_progress_bar()
            self.trigger_update()

    def on_prediction_basis_changed(self):
        if self.current_row is not None and self.current_category is not None:
            self.last_chart_data = {}
            self.show_progress_bar()
            self.trigger_update()

    def open_search_ticker_dialog(self):
        dialog = SearchTickerDialog(self)
        dialog.exec_()

    def open_trade_dialog(self, category, row):
        if not self.current_user:
            QMessageBox.warning(self, "Error", "Please login to trade.")
            return
        dialog = TradeDialog(self, category, row)
        dialog.exec_()

    def open_login_dialog(self):
        if self.current_user:
            QMessageBox.information(self, "Info", "Already logged in.")
            return
        dialog = AuthDialog(self)
        if dialog.exec_():
            self.portfolio_tab.withdraw_button.setEnabled(True)

    def open_register_dialog(self):
        dialog = AuthDialog(self)
        dialog.setCurrentTab("Register")
        dialog.exec_()

    def update_portfolio(self):
        self.portfolio_tab.update_portfolio()

    def trigger_update(self):
        if (datetime.now() - self.last_update_time).total_seconds() < 5:
            logger.debug("Skipping update: too frequent")
            return
        self.last_update_time = datetime.now()
        logger.debug("Triggering update")
        self.show_progress_bar()
        self.update_triggered.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockMarketApp()
    window.show()
    sys.exit(app.exec_())