#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:39:44 2024

@author: Julien
"""

import dash
from dash import dcc,html
from dash.dependencies import Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from lib import monteCarloSimul  
from lib import run_CPPI  
import webbrowser
import numpy as np 

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("CPPI With Monte-Carlo Simulations"),
    html.Label("Number of Scenarios"),
    dcc.Slider(
        id='scenario-slider',
        min=0,
        max=1000,
        value=400,
        marks={0:'0', 250: '250', 500:'500', 750:'750', 1000: '1000'},
        step=50,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Return (μ)"),
    dcc.Slider(
        id='mu-slider',
        min=-0.01,
        max=0.11,
        value=0.08,
        marks = {-0.01:'-1%', 0.02: '2%', 0.05:'5%',  0.08:'8%', 0.11: '11%'},
        step=0.01,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Volatility (σ)"),
    dcc.Slider(
        id='sigma-slider',
        min=0.01,
        max=0.25,
        value=0.17,
        marks={0.01: '1%', 0.05 : '5%', 0.09: '9%', 0.13: '13%', 0.17 : '17%',
               0.21: '21%', 0.25: '25%'},
        step=0.01,
        tooltip={"placement": "bottom", "always_visible": True}
        
    ),
    html.Label("Multiplier (m)"),
    dcc.Slider(
        id='m-slider',
        min=1,
        max=10,
        value=3,
        marks={1:'1', 3:'3', 5:'5', 8:'8', 10:'10'},
        step=1,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Floor"),
    dcc.Slider(
        id='floor-slider',
        min=0.0,
        max=1,
        value=0.75,
        marks={0.0 :'0%', 0.25: '25%', 0.5 : '50%', 0.75 : '75%', 1 : '100%'},
        step=0.05,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Risk-Free Rate (Annual)"),
    dcc.Slider(
        id='rf-rate-slider',
        min=0.01,
        max=0.1,
        value=0.03,
        marks={0.01 : '1%', 0.025 : '2.5%', 0.05 : '5%', 0.075: '7.5%', 
               0.1 : '10%'},
        step=0.005,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
   
    dcc.Graph(id='cppi-graph')
])

@app.callback(
    Output('cppi-graph', 'figure'),
    [
        Input('scenario-slider', 'value'),
        Input('mu-slider', 'value'),
        Input('sigma-slider', 'value'),
        Input('m-slider', 'value'),
        Input('floor-slider', 'value'),
        Input('rf-rate-slider', 'value')
    ]
)
def update_graph(scenario, mu, sigma, m, floor, riskfree_rate):
   
    # Default start value
    start = 1000
    
    # Simulate GBM returns and convert into a DataFrame
    sim_rets = monteCarloSimul(n_scenarios=scenario, mu=mu, sigma=sigma, steps_per_year=12, prices=False)
    risky_r = pd.DataFrame(sim_rets)
    
    # Run CPPI
    cppi = run_CPPI(risky_r, riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    
    # Extract wealth
    wealth = cppi['Wealth']
    
    # Create subplots 
    fig = make_subplots(rows=1, cols=2,shared_yaxes=True,
                        horizontal_spacing=0.006)
    
    # Plot Monte-Carlo CPPI  
    for c in wealth.columns : 
        fig.add_trace(go.Scatter(x=wealth.index,
                                 marker={'color': 'indianred'},
                                 opacity=0.7,
                                 y=wealth[c]
                                 ),
                                 row=1,
                                 col=1,
                                 )
    # Plot the protected Floor
    floor_plot = [start*floor for i in range(wealth.shape[0])]
    floor_name = ['Floor' for i in range(wealth.shape[0])]
    fig.add_trace(go.Scatter(y=floor_plot,
                             marker={'color':'black'
                                     },
                             text=floor_name,
                             ),
                             row=1,
                             col=1
                             )
    
    # Plot terminal values of CPPI 
    fig.add_trace(go.Histogram(y=wealth.iloc[-1],
                         orientation='h', 
                         marker={'color': 'indianred'},
                         opacity=0.7,
                         name='Terminal Values'),
                         row=1, 
                         col=2
                         )
    
    # Plot the Median and the mean of the terminal values 
    l_median = [wealth.iloc[-1].median()for i in range(wealth.shape[0])]
    l_mean = [wealth.iloc[-1].mean() for i in range(wealth.shape[0])]
    l_median_name = ['Median' for i in range(wealth.shape[0])]
    l_mean_name = ['Mean' for i in range(wealth.shape[0])]
    fig.add_trace(go.Scatter(y=l_median,
                             marker={'color':'green'
                                     },
                             text=l_median_name
                             ),
                             row=1,
                             col=2
                             )
    fig.add_trace(go.Scatter(y=l_mean,
                             marker={'color':'blue'
                                     },
                             text=l_mean_name
                             ),
                             row=1,
                             col=2
                             )
    
    # Update figures 
    fig.update_layout(showlegend=False,margin=dict(l=40, r=40, t=40, b=40),
                      font=dict(color='black'),
                  )
    
    # Compute the number of violations 
    failure = np.less(wealth.iloc[-1],floor*start)
    failure = sum(failure)
    
    # Add annotation to the second subplot (row=1, col=2)
    fig.add_annotation(
        x=50,  
        y=np.max(wealth.iloc[-1]),  
        xref='x2',  
        yref='y2',  
        text=f'Violations: {failure}',  
       font=dict(size=20),  
       showarrow=False,
)
    fig.add_annotation(
        x=50,  
        y=np.max(wealth.iloc[-1]-1000),  
        xref='x2',  
        yref='y2',  
        text=f'Mean: {round(l_mean[0])} CHF',  
       font=dict(size=20,color='blue'),  
       showarrow=False,
)
    fig.add_annotation(
        x=50,  
        y=np.max(wealth.iloc[-1]-2000),  
        xref='x2',  
        yref='y2',  
        text=f'Median: {round(l_median[0])} CHF',  
       font=dict(size=20, color='green'),  
       showarrow=False,
)
    
    

    fig.update_xaxes(title_text='Months (10 years)', row=1,col=1)
    fig.update_yaxes(title_text='Values', row=1,col=1)
    fig.update_xaxes(title_text='Frequence', row=1,col=2)

    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
    webbrowser.open('http://127.0.0.1:8050/')
