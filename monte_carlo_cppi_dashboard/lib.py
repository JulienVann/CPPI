#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:44:38 2024

@author: Julien
"""
import numpy as np 
import pandas as pd 

def monteCarloSimul(n_years = 10, n_scenarios=1000, 
                    mu=0.07, sigma=0.15, steps_per_year=12, 
                    s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def run_CPPI(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    """
    
    CPPI Simulation 
    :risky_r: Risky assets 
    :safe_r: safe asset e.g. treasury bills 
    :m: multiplier 
    :start: Initial Wealth 
    :floor: protected floor 
    :riskfree_rate: safe asset if no safe asset is provided 
    :return: a dictionary with cppi data 

    """
    
    # Computations of the parameters 
    dates = risky_r.index 
    n_steps = len(dates)
    account_value = start 
    floor_value = floor * start 
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r,columns=['R'])
    
    # In case no safe asset provided 
    if safe_r is None :
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate/12 # Monthly data 

    # Keep track of the history 
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    floor_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps) : 
        
        # Cushion 
        cushion = (account_value - floor_value)/account_value

        # Risky asset 
        risky_w = m * cushion 

        # Avoid leverage 
        risky_w = np.minimum(risky_w,1)

        # Avoid being negative 
        risky_w = np.maximum(risky_w,0)

        # Safe asset 
        safe_w = 1 - risky_w

        # Allocation in the risky assets 
        risky_alloc = risky_w * account_value

        # Allocation in the riskless assets 
        safe_alloc = safe_w * account_value 

        # Update account value 
        account_value = risky_alloc * (1+risky_r.iloc[step]) + safe_alloc * (1+safe_r.iloc[step])

        # Keep track of the history 
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floor_history.iloc[step] = floor_value
        
        # Compute the risky wealth 
        risky_wealth = start * (1 + risky_r).cumprod()
        
    return {
        'Wealth':account_history,
        'Risky wealth':risky_wealth,
        'Floor value': floor_history,
        'Risk budget':cushion_history,
        'Risky allocation': risky_w_history,
        'm':m,
        'start':start,
        'floor':floor,
        'risky_r':risky_r,
        'safe_r':safe_r
        }


















