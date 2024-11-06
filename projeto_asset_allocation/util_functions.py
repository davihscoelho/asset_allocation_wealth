import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_total_performance(retorno,w): 

    """"
    Parameters:
    rets: Pd series rets
    """
    rets_port = np.dot(retorno.loc[:,w.index],w)
    ret = round(np.exp(rets_port.sum())-1,2)
    return ret

def calculate_cagr(retorno,w):

    n = len(retorno)
    ret = calculate_total_performance(retorno,w)
    cagr = ret**(12/n)

    return cagr

def calculate_rets_rolante(retorno,w):
    rets_port = np.dot(retorno.loc[:,w.index],w)
    ret = pd.Series(rets_port).rolling(window=12).sum()

    return ret

def calculate_rets_rolante_media(retorno,w):
    rets_port = np.dot(retorno.loc[:,w.index],w)
    ret = pd.Series(rets_port).rolling(window=12).sum()
    

    return ret.mean()

def calculate_vol_portfolio(retorno,weight):

    cov_matrix = retorno.cov()

    vol_portfolio = np.dot(weight.T,np.dot(weight,cov_matrix))

    vol_portfolio = np.sqrt(vol_portfolio)*np.sqrt(12)

    return round(vol_portfolio,4)

def calculate_sharpe_ratio(retorno, w,risk_free_rate=0.0):
    """
    Calcula o Sharpe Ratio.

    Parâmetros:
    returns (pd.Series): Série de retornos.
    risk_free_rate (float): Taxa livre de risco (pode ser anualizada).

    Retorna:
    float: Sharpe Ratio.
    """
    rets_port = np.dot(retorno.loc[:,w.index],w)
    vol = calculate_vol_portfolio(retorno,w)
    excess_returns = rets_port.mean()*12 - risk_free_rate
    return excess_returns / vol

def calculate_skewness(retorno,w):
    """
    Calcula a Skewness.

    Parâmetros:
    returns (pd.Series): Série de retornos.

    Retorna:
    float: Skewness.
    """
    rets_port = np.dot(retorno.loc[:,w.index],w)
    return skew(rets_port)

def calculate_kurtosis_value(retorno,w):
    """
    Calcula a Kurtosis.

    Parâmetros:
    returns (pd.Series): Série de retornos.

    Retorna:
    float: Kurtosis.
    """
    rets_port = np.dot(retorno.loc[:,w.index],w)
    return kurtosis(rets_port)

def calculate_var(retorno,w):
  rets_port = np.dot(retorno.loc[:,w.index],w)
  return np.percentile(rets_port,5)

def calculate_cvar(retorno,w):
  rets_port = np.dot(retorno.loc[:,w.index],w)
  return np.mean(rets_port[rets_port<=np.percentile(rets_port,5)])

def calculate_mdd(retorno,w):
    rets_port = np.dot(retorno.loc[:,w.index],w)
    cum_rets = np.exp(rets_port.cumsum())-1
    prices_d0 = 100
    prices = np.linspace(1,len(rets_port)+1,len(rets_port)+1)
    prices[0] = prices_d0
    
    prices[1:] = 100 * (1+cum_rets)
    df = pd.DataFrame(prices, columns=['Value'])    
    
    df['Peak'] = df['Value'].cummax()  # Cumulative max (peak up to that point)
    df['Drawdown'] = (df['Value'] - df['Peak']) / df['Peak']  # Drawdown calculation
    df['Drawdown_Percent'] = df['Drawdown']
    max_drawdown = df['Drawdown'].min()
    drawdown_start = df['Drawdown'].idxmin()  # Day when the maximum drawdown occurs
    
    drawdown_end_indices = np.where(df['Value'][drawdown_start:].values >= df['Peak'][drawdown_start])[0]
    # Check if there is a valid drawdown_end, otherwise set it to None
    if len(drawdown_end_indices) > 0:
        drawdown_end = drawdown_start + drawdown_end_indices[0]  # Convert to the original index
    else:
        drawdown_end = None  # Handle case where drawdown doesn't recover
    
    # Calculate drawdown length if valid
    drawdown_length = (drawdown_end - drawdown_start) if drawdown_end is not None else None

    return max_drawdown,drawdown_length

def getting_metrics (retorno,w):
    metrics = {
    "Total Return": f"{(calculate_total_performance(retorno,w)):.1%}",
    "CAGR": f"{(calculate_cagr(retorno,w)-1):.1%}",
    "AVG R.W 12MM": f"{(calculate_rets_rolante_media(retorno,w)):.1%}",
    "Vol a.a": f"{(calculate_vol_portfolio(retorno,w)):.1%}",
    "MDD": f"{(calculate_mdd(retorno,w)[0]):.1%}",
    "MDD Recovery Time": calculate_mdd(retorno,w)[1],
    "VAR": f"{(calculate_var(retorno,w)):.1%}",
    "CVAR": f"{(calculate_cvar(retorno,w)):.1%}",
    "Skewness": f"{(calculate_skewness(retorno,w)):.1f}",
    "Kurtosi": f"{(calculate_kurtosis_value(retorno,w)):.1f}",
    "Sharpe": f"{(calculate_sharpe_ratio(retorno,w)):.2f}"

    }

    return pd.DataFrame(metrics,index=[0])