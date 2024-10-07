import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import plotly.graph_objs as go
import plotly.express as px


def get_baseline_positions(n, baseline_min, baseline_max):

    baseline = np.zeros((1,5))


    baseline[0][0] = random.randint(baseline_min, baseline_max)
    baseline[0][1] = random.randint(baseline_min, baseline_max)
    baseline[0][2] = random.randint(baseline_min, baseline_max)
    baseline[0][3] = random.randint(baseline_min, baseline_max)
    baseline[0][4] = random.randint(baseline_min, baseline_max)

    '''
    baseline[0][0]  = random.randint(-100,100)
    baseline[0][1] = -baseline[0][0]
    baseline[0][2] = random.randint(-100,100)
    baseline[0][3] = -baseline[0][2]
    baseline[0][4] = random.randint(-100,100)
    '''

    #print("Initial Baseline", baseline)


    # there are C(5,2) = 5!/(5-2)!(2!) = 10 unique pairs for the exchange rates


    exchange_rate_1_2 = 2
    exchange_rate_1_3 = 0.5
    exchange_rate_1_4 = 1.5
    exchange_rate_1_5 = 3

    exchange_rate_2_3 = 0.5
    exchange_rate_2_4 = 1.25
    exchange_rate_2_5 = 0.75

    exchange_rate_3_4 = 2
    exchange_rate_3_5 = 0.5

    exchange_rate_4_5 = 1.5
    
    exchange_rates = np.zeros((5,5))

    for i in range(exchange_rates.shape[0]):
        for j in range(exchange_rates.shape[1]):
            
            if i == j:
                exchange_rates[i][j] = 1  # Set diagonal to 1 (since the exchange rate to itself is 1)
            else:
                # Assign the exchange rates based on (i,j) index
                if i == 0 and j == 1:
                    exchange_rates[i][j] = exchange_rate_1_2
                elif i == 0 and j == 2:
                    exchange_rates[i][j] = exchange_rate_1_3
                elif i == 0 and j == 3:
                    exchange_rates[i][j] = exchange_rate_1_4
                elif i == 0 and j == 4:
                    exchange_rates[i][j] = exchange_rate_1_5
                elif i == 1 and j == 2:
                    exchange_rates[i][j] = exchange_rate_2_3
                elif i == 1 and j == 3:
                    exchange_rates[i][j] = exchange_rate_2_4
                elif i == 1 and j == 4:
                    exchange_rates[i][j] = exchange_rate_2_5
                elif i == 2 and j == 3:
                    exchange_rates[i][j] = exchange_rate_3_4
                elif i == 2 and j == 4:
                    exchange_rates[i][j] = exchange_rate_3_5
                elif i == 3 and j == 4:
                    exchange_rates[i][j] = exchange_rate_4_5
                else:
                    # For reverse exchange rates (j to i), set as reciprocal
                    exchange_rates[i][j] = 1 / exchange_rates[j][i]

    transactions = np.zeros((n,5))

    for i in range(n):

        currency = random.randint(0,4)
        remaining_numbers = [x for x in range(5) if x!= currency]
        counter_currency = random.choice(remaining_numbers)

        buy_or_sell = random.randint(0,1)

        # buy_or_sell = 1 <=> we buy the currency
        # buy_or_sell = 0 <=> we sell the currency
        
        if buy_or_sell == 0:
            transactions[i,currency] = random.randint(5,10)
            transactions[i, counter_currency] = -transactions[i,currency]*exchange_rates[currency][counter_currency]
        elif buy_or_sell == 1:
            transactions[i,currency] = random.randint(-10,-5)
            transactions[i, counter_currency] = transactions[i,currency]*exchange_rates[currency][counter_currency]*(-1)


    positions = np.vstack([baseline, transactions])

    #print(positions)

    return positions



def SettlementRisk(position):
    return np.sum(np.maximum(0,position))

# %% [markdown]
# #### **Step 4:** Define our strategies
# ---

# %%
def BruteForce(positions):

    bf_list = []
    total_position = np.zeros((1,5))
    
    
    total_position = np.vstack([total_position, positions[0,:]])
    total = SettlementRisk(total_position)
    bf_list.append(total)

    for i in range(positions.shape[0]-1):
        total_position = np.vstack([total_position, positions[i+1,:]])

        temp =  np.sum(total_position, axis=0)
        total = SettlementRisk(temp)
        bf_list.append(total)
    
    #total_position = np.sum(positions, axis=0)

    return [SettlementRisk(total), bf_list]


def Linearisation(positions):

    lin_list = []
    baseline = positions[0,:]

    baseline_exposure = SettlementRisk(baseline)

    total = baseline_exposure
    lin_list.append(total)

    for i in range(positions.shape[0]-1):
        decoupled = np.vstack([baseline, positions[i+1,:]])
        decoupled = np.sum(decoupled, axis=0)
                
        decoupled_exposure = SettlementRisk(decoupled)
        total += decoupled_exposure - baseline_exposure
        lin_list.append(total)

    return [total, lin_list]

def Conservative(positions):

    total = 0
    cons_list = []
    for i in range(positions.shape[0]):

        total += SettlementRisk(positions[i,:])
        cons_list.append(total)

    return [total, cons_list]

def Averages(positions):
    avg_list = []
    baseline = positions[0,:]
    n = positions.shape[0]-1
    baseline_exposure = SettlementRisk(baseline)

    total = baseline_exposure
    avg_list.append(total)

    for i in range(positions.shape[0]-1):

        decoupled = np.vstack([baseline, positions[i+1,:]])
        decoupled[1] *= n
        decoupled = np.sum(decoupled, axis=0)
        decoupled_exposure = SettlementRisk(decoupled)
        total += 1/n*(decoupled_exposure - baseline_exposure)
        avg_list.append(total)

    return [total, avg_list]

def sim(m, baseline_min, baseline_max):

    data = get_baseline_positions(m, baseline_min, baseline_max)

    [total_BF, bf_list] = BruteForce(data)
    [total_Linearisation, lin_list] = Linearisation(data)
    [total_Conservative, cons_list] = Conservative(data)
    [total_Averages, avg_list] = Averages(data)

    x = np.linspace(0, m, m+1)

    # Create a figure for the Exposure (Settlement Risk) comparison
    fig_exposure = go.Figure()

    # Add traces for each method
    fig_exposure.add_trace(go.Scatter(x=x, y=bf_list, mode='lines', name='Brute Force (Batch)', line=dict(color='blue')))
    fig_exposure.add_trace(go.Scatter(x=x, y=lin_list, mode='lines', name='Linearisation', line=dict(color='green')))
    fig_exposure.add_trace(go.Scatter(x=x, y=avg_list, mode='lines', name='Averages', line=dict(color='yellow')))
    fig_exposure.add_trace(go.Scatter(x=x, y=cons_list, mode='lines', name='Conservative', line=dict(color='red')))

    # Customize layout
    fig_exposure.update_layout(
        title="Exposure (Settlement Risk) Comparison",
        xaxis_title="Trade Number",
        yaxis_title="Exposure (Settlement Risk)",
        legend_title="Methods",
        grid=dict(show=True),
        template="plotly_white"  # Optional: adds a white background grid style
    )

    # Display the interactive figure
    fig_exposure.show()

    # Calculate the percentage difference between Linearisation and Brute Force
    diff_lin_bf = [(lin_list[i] - bf_list[i]) / bf_list[i] * 100 for i in range(len(lin_list))]
    min_diff = np.min(diff_lin_bf)
    max_diff = np.max(diff_lin_bf)

    # Create a figure for the percentage difference plot
    fig_diff = go.Figure()

    # Add a trace for the percentage difference
    fig_diff.add_trace(go.Scatter(x=x, y=diff_lin_bf, mode='lines', line=dict(color='red')))

    # Customize layout
    fig_diff.update_layout(
        title="Percentage Difference between Linearisation and Brute Force",
        xaxis_title="Trade Number",
        yaxis_title="Percentage Difference",
        grid=dict(show=True),
        template="plotly_white"
    )

    # Display the interactive figure
    fig_exposure.show(renderer="browser")  # This will open the graph in your default browser
    fig_diff.show(renderer="browser")

    # Print the range of percentage differences
    print("Range of values for percentage difference: [" + str(min_diff) + ", " + str(max_diff) + "]")
    print("\n")
    print("\n")
    print("NOTE: Positive percentage implies linearisation is working correctly (overestimating)")
    print("      Negative percentage implies linearisation is optimistic (underestimating exposure)")
