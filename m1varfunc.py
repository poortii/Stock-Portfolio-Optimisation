import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import seaborn as sns
import matplotlib.pyplot as plt

def plot_trends(stock_data):
    """Plot interactive stock price trends"""
    melted_data = stock_data.melt(id_vars='Date', var_name='Company', value_name='Price')
    fig = px.line(
        melted_data, 
        x='Date', 
        y='Price', 
        color='Company',
        title='Stock Price Trends',
        labels={'Price': 'Price (₹)', 'Date': ''}
    )
    fig.update_layout(
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def covariance_matrix(portfolio):
    """Calculate annualized sample covariance matrix"""
    return risk_models.sample_cov(portfolio, frequency=252)

def heatmap_covar(cov_matrix):
    """Create interactive correlation heatmap"""
    fig = px.imshow(
        cov_matrix,
        x=cov_matrix.columns,
        y=cov_matrix.index,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Stock Correlation Matrix'
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        coloraxis_colorbar=dict(title='Correlation')
    )
    return fig

def return_capm(portfolio):
    """Calculate expected returns using CAPM"""
    return expected_returns.capm_return(portfolio)

def max_sharpe_weights(mu, sigma, risk_free_rate=0.025):
    ef = EfficientFrontier(mu, sigma)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    
    # Convert OrderedDict to numpy array in correct order
    weights_array = np.array([cleaned_weights[ticker] for ticker in mu.index])
    expected_return = np.dot(mu, weights_array)
    volatility = np.sqrt(np.dot(weights_array, np.dot(sigma, weights_array)))
    sharpe = (expected_return - risk_free_rate) / volatility
    
    return cleaned_weights, (expected_return, volatility, sharpe)

def min_variance_weights(mu, sigma):
    ef = EfficientFrontier(mu, sigma)
    ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    
    # Convert OrderedDict to numpy array
    weights_array = np.array([cleaned_weights[ticker] for ticker in mu.index])
    expected_return = np.dot(mu, weights_array)
    volatility = np.sqrt(np.dot(weights_array, np.dot(sigma, weights_array)))
    sharpe = expected_return / volatility
    
    return cleaned_weights, (expected_return, volatility, sharpe)

def plot_weights(weights):
    """Create interactive portfolio allocation chart"""
    weights_series = pd.Series(weights).sort_values(ascending=True)
    fig = px.bar(
        weights_series,
        orientation='h',
        labels={'index': 'Stock', 'value': 'Weight'},
        title='Portfolio Allocation',
        color=weights_series.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title='',
        xaxis_title='Allocation %',
        coloraxis_showscale=False
    )
    fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
    return fig


def plot_efficient_frontier(mu, sigma, risk_free_rate=0.025):
    ef = EfficientFrontier(mu, sigma)
    fig = go.Figure()
    
    # Generate random portfolios
    n_samples = 1000
    random_weights = np.random.dirichlet(np.ones(len(mu)), n_samples)
    
    # Calculate returns and volatilities
    rets = [np.dot(mu, w) for w in random_weights]
    vols = [np.sqrt(np.dot(w, np.dot(sigma, w))) for w in random_weights]
    
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode='markers',
        name='Random Portfolios',
        marker=dict(color='lightgray', size=5)
    ))
    
    # Efficient frontier points
    ret_range = np.linspace(min(rets), max(rets)*0.9, 20)
    vol_range = []
    for r in ret_range:
        try:
            ef.efficient_return(r)
            w = np.array([ef.clean_weights()[ticker] for ticker in mu.index])
            vol_range.append(np.sqrt(np.dot(w, np.dot(sigma, w))))
        except:
            continue
    
    fig.add_trace(go.Scatter(
        x=vol_range, y=ret_range, mode='lines',
        name='Efficient Frontier',
        line=dict(color='royalblue', width=3)
    ))
    
    # Optimal portfolios
    ef.max_sharpe(risk_free_rate)
    w_sharpe = np.array([ef.clean_weights()[ticker] for ticker in mu.index])
    fig.add_trace(go.Scatter(
        x=[np.sqrt(np.dot(w_sharpe, np.dot(sigma, w_sharpe)))],
        y=[np.dot(mu, w_sharpe)],
        mode='markers+text',
        name='Max Sharpe',
        marker=dict(size=15, color='green')
    ))
    
    ef.min_volatility()
    w_minvol = np.array([ef.clean_weights()[ticker] for ticker in mu.index])
    fig.add_trace(go.Scatter(
        x=[np.sqrt(np.dot(w_minvol, np.dot(sigma, w_minvol)))],
        y=[np.dot(mu, w_minvol)],
        mode='markers+text',
        name='Min Volatility',
        marker=dict(size=15, color='red')
    ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Return'
    )
    return fig