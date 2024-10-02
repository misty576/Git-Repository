import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

# Columns correspond to CURRENCIES
# Column 1: USD
# Column 2: GBP
# Column 3: EUR
# Column 4: AUD
# Column 5: JPY


def get_baseline_positions(n):

    baseline = np.zeros((1,5))

    '''
    baseline[0][0] = 100
    baseline[0][1] = -100
    baseline[0][2] = 50
    baseline[0][3] = -50
    baseline[0][4] = 200
    '''
    baseline[0][0] = 100
    baseline[0][1] = -100
    baseline[0][2] = 50
    baseline[0][3] = -50
    baseline[0][4] = 20

    #print("Initial Baseline", baseline)


    # there are C(5,2) = 5!/(5-2)!(2!) = 10 unique pairs for the exchange rates

    '''
    exchange_rate_1_2 = 0.9
    exchange_rate_1_3 = 1.1
    exchange_rate_1_4 = 0.8
    exchange_rate_1_5 = 1.2

    exchange_rate_2_3 = 0.75
    exchange_rate_2_4 = 1.25
    exchange_rate_2_5 = 0.85

    exchange_rate_3_4 = 1.3
    exchange_rate_3_5 = 0.6

    exchange_rate_4_5 = 1.4

    '''
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

        if buy_or_sell == 0:
            transactions[i,currency] = random.randint(5,10)
            transactions[i, counter_currency] = -transactions[i,currency]*exchange_rates[currency][counter_currency]
        elif buy_or_sell == 1:
            transactions[i,currency] = random.randint(-10,-5)
            transactions[i, counter_currency] = transactions[i,currency]*exchange_rates[currency][counter_currency]*(-1)


    positions = np.vstack([baseline, transactions])

    print(positions)

    return positions



def SettlementRisk(position):
    return np.sum(np.maximum(0,position))

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




#positions = get_baseline_positions(5)



'''
print("Total Settlement Risk for Brute Force: ", BruteForce(positions))
print("Total Settlement Risk for Linearisation: ", Linearisation(positions))
print("Total Settlement Risk for Conservative: ", Conservative(positions))
print("Total Settlement Risk for Averages: ", Averages(positions))
'''


# TAKE A SAMPLE DATASET AND MAKE SURE THAT IT IS CALCULATING STUFF CORRECTLY!!


def sim(m):

    data = get_baseline_positions(m)

    
    '''
    data = np.array([[ 10., -10.,  5.,  -5.,  20. ],
                    [ -5.,   0.,  2.5,  0.,   0.  ],
                    [  7., -14.,  0.,   0.,   0.  ],
                    [  0.,   0.,  6., -12.,  0.  ],
                    [  0.,   0., 10.,  0.,  -5.  ],
                    [  0.,   8.,  0., -10.,  0.  ]])
    '''

    [total_BF, bf_list] = BruteForce(data)
    [total_Linearisation, lin_list] = Linearisation(data)
    [total_Conservative, cons_list] = Conservative(data)
    [total_Averages, avg_list] = Averages(data)

    print("Brute Force: ", total_BF)
    print("\n")
    print("Linearisation: ", total_Linearisation)
    print("\n")
    print("Conservative: ", total_Conservative)
    print("\n")
    print("Averages: ", total_Averages)


    x = np.linspace(0,m,m+1)

    plt.figure()
    plt.plot(x,bf_list, label="Brute Force (Batch)", color= "b")
    plt.plot(x,lin_list, label="Linearisation", color= "g")
    plt.plot(x,avg_list, label="Averages", color= "y")
    plt.plot(x,cons_list, label="Conservative", color= "r")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Trade Number")
    plt.ylabel("Exposure (Settlement Risk)")
    plt.show()

sim(10)