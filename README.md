# CPPI
Monte Carlo Simulation with CCPI Strategy Dashboard

## Overview
This project demonstrates a Monte Carlo simulation of portfolio returns using a Constant Proportion Portfolio Insurance (CPPI) strategy, along with a Dash web dashboard to visualize the simulation results and distribution of terminal values.

## Installation Instructions
Follow these steps to set up and run the Monte Carlo simulation dashboard on your local machine:

1. Clone the project repository from GitHub:
git clone <https://github.com/JulienVann/CPPI.git>

2. Open the directory:
cd monte_carlo_cppi_dashboard

3. Install Required Python Packages:
pip install -r requirements.txt

4. Run the Dash Application:
python dash_cppi.py

5. Open a Web Browser:
Navigate to http://127.0.0.1:8050 to view the application.

## Usage
The web dashboard consists of two plots:

- Left Plot (Simulated CPPI Strategy):
Shows the portfolio value over time using the simulated CPPI strategy.

- Right Plot (Distribution of Terminal Values):
Displays the distribution of terminal values resulting from the CPPI Strategy.

Interact with the dashboard:
Adjust parameters (e.g., risk level, initial allocation, floor level) to observe the impact on the simulated CPPI strategy and terminal value distribution.
