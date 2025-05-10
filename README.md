# 📈 Stock Portfolio Optimisation

This project predicts stock prices using time series forecasting models and applies portfolio optimization techniques to maximize returns or minimize risk. The goal is to help investors make data-driven decisions by combining machine learning and quantitative finance.

---

## 🚀 Features

- 📊 **Stock Price Forecasting**: Forecast future stock prices using techniques like:
  - LSTM (Long Short-Term Memory)
  - VAR (Vector Auto Regression)
  - Other ML-based models
- 📈 **Portfolio Optimization**:
  - Maximize returns given a target risk (Markowitz-style)
  - Minimize risk for a given return
  - Simulate and backtest optimized portfolios
- 📉 **Risk Metrics**:
  - Value-at-Risk (VaR)
  - Sharpe Ratio
  - Drawdown tracking

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas**, **NumPy**, **Matplotlib**
- **Scikit-learn**
- **TensorFlow / Keras** (for LSTM models)
- **Statsmodels** (for VAR modeling)

---

## 📁 Project Structure

```bash
stock-price-optimisation/
├── m1varfunc.py                # VAR model functions
├── m2var.py                    # Additional model/analysis script
├── stock_prediction_func.py    # Main forecasting functions
├── stock_price_env/            # Virtual environment (ignored via .gitignore)
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

### ⚙️ Setup Instructions
Clone the repo:

```bash

git clone https://github.com/poortii/Stock-Portfolio-Optimisation.git
cd Stock-Portfolio-Optimisation

```
### Create virtual environment:

```bash 
python -m venv stock_price_env
Activate the environment:
```
On Windows:
```bash 
stock_price_env\Scripts\activate
```
On macOS/Linux:

```bash

source stock_price_env/bin/activate
```
Install dependencies:

```bash

pip install -r requirements.txt
```
Run your script:

```bash

python stock_prediction_func.py
```

###📌 Notes
Make sure to have historical stock data ready as input.

You can customize the models or constraints in m1varfunc.py and m2var.py.

Avoid uploading the virtual environment to GitHub — it's already ignored in .gitignore.

###📬 Contact
Made by @poortii — feel free to reach out for collaboration or questions!
