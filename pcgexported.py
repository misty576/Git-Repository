# %%
# Pre-settlement risk is the possibility that one party in a contract will fail to meet its obligations under that contract, resulting in a default before the settlement date. 
# mark to market (MTM) - The present value of all the payments that a party is expecting to receive, less those it is obliged to make
# ctb = contribution
# there will also be BASELINE RESETS = the exposure for the orginal baseline will be recalculated due to changing market conditions


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

# We define the exposure (MTM) calculator
def currentExposure(mtm):
    return max(0,mtm)

# We calculate the notional
def addOn(notional, addOnFactor):
    return notional*addOnFactor

# The total exposure is the sum of the two above functions
def totalExposure(mtm, notional, addOnFactor):
    currExp = currentExposure(mtm)
    currAddOn = addOn(notional, addOnFactor)

    return currExp + currAddOn

# The brute force takes all the impacts, sums them together, and calculates the bulk exposure.
def psrBruteForce(position, impacts, addOnFactor):
    
    # For each step, we sum all the impacts into a variable called 'total' and calculate the exposure.

    totalMtm = position[0] + np.sum(impacts[:,0])
    totalNotional = position[1] + np.sum(impacts[:,1])
    return totalExposure(totalMtm, totalNotional, addOnFactor)

def psrConservative(position, impacts, addOnFactor):

    # For each step, we calculate the exposure individually for each impact and add that to the total.
    
    total = 0
    total += totalExposure(position[0], position[1], addOnFactor)

    for i in range(len(impacts)):
        total += totalExposure(impacts[i,0], impacts[i,1], addOnFactor)

    return total 

def psrLinearisation(position, impacts, addOnFactor):

    # We take the baseline position and for each subsequent impact, we take that exposure and 
    # subtract it from the original baseline position. (Read MXWiki for more detail)

    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure

    for i in range(len(impacts)):
        total += totalExposure(position[0]+impacts[i,0], position[1]+impacts[i,1], addOnFactor) - position_exposure

    return max(0,total)

def psrAverages(position, impacts, addOnFactor, n):
    
    # This method is very similar to the linearisation approach.
    
    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure

    for i in range(len(impacts)):
        total += 1/n*(totalExposure(position[0]+n*impacts[i,0], position[1]+n*impacts[i,1], addOnFactor) - position_exposure)
    
    return total



def remove_outliers(data, threshold = 2):

    # The growth in exposure from the baseline can be exceptionally high for a small number of trades (since we are going from 0% difference to, for instance, a 20% differnce depending on the impacts we use).
    # This function removes the outliers that impact our analysis of the graph, by using standard mean and variance approaches

    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    cleaned_data = data.copy()

    for i in range(len(data)):

        if -(mean+threshold*std_dev) < data[i] < (mean+threshold*std_dev):
            cleaned_data[i] = data[i]
        else:
            if i == 0:
                neighbors = [data[j] for j in range(max(0, i+5), min(len(data), i+3)) if j != i]
                cleaned_data[i] = np.median(neighbors)
            elif i == len(data) - 1:
                continue
            else:
                # Method: use median and neighbours
                neighbors = [data[j] for j in range(max(0, i-2), min(len(data), i+3)) if j != i]
                cleaned_data[i] = np.median(neighbors)
    return cleaned_data
    

def exposure_simulation(n, baselineMTM, baselineNotional, mtmMin, mtmMax, notionalMin, notionalMax, addOnWidget, lin=True, cons=False, avg=False):

    # here we want to obtain a graph showing the growth of the difference between baseline/brute force as number of trades increase
    addOnVal = 0.005
    baseline_position = [baselineMTM, baselineNotional]  

    # Here, for each number n, we will take a new dataset (baseline and impacts) and calculate the exposure
    # for each strategy by adding them to each respective list.
    
    bf_list = []
    cons_list = []
    lin_list = []
    avg_list = []

    for i in range(1,n):
    
        mtm_notional_matrix = np.zeros((i,2)) # Column 1 will be MTM, Column 2 will be Notional

        for j in range(i):
            mtm_notional_matrix[j,0] = random.randint(mtmMin,mtmMax)

        for j in range(i):
            mtm_notional_matrix[j,1] = random.randint(notionalMin, notionalMax)


        bf_list.append(psrBruteForce(baseline_position, mtm_notional_matrix, addOnVal))
        cons_list.append(psrConservative(baseline_position, mtm_notional_matrix, addOnVal))
        lin_list.append(psrLinearisation(baseline_position, mtm_notional_matrix, addOnVal))
        avg_list.append(psrAverages(baseline_position, mtm_notional_matrix, addOnVal, n))


    # We calculate the percentage difference from the brute force approach using these lists
    diff_cons = [(cons_list[i]-bf_list[i])/bf_list[i] * 100 for i in range(len(bf_list))]
    diff_lin = [(lin_list[i]-bf_list[i])/bf_list[i] * 100 for i in range(len(bf_list))]
    diff_avg = [(avg_list[i]-bf_list[i])/bf_list[i] * 100 for i in range(len(bf_list))]


    # We then cleanse the dataset of its outliers
    diff_cons = remove_outliers(diff_cons, 3)
    diff_lin = remove_outliers(diff_lin, 3)
    diff_avg = remove_outliers(diff_avg, 3)
    
    
    #####################   POST-PROCESSING   #####################

    x = np.arange(1,n)

    plt.figure(figsize=(7, 3))
    plt.grid(True)

    if lin == True:
        plt.plot(x[int(n/100):],diff_lin[int(n/100):],'r', label="Linearisation")
    
    if cons == True:
        plt.plot(x,diff_cons,'b', label="Conservative")

    if avg == True:
        plt.plot(x,diff_avg,'g', label="Averages")

    plt.xlabel("# of trades")
    plt.ylabel("% Diff. from Brute Force")
    plt.legend()
    plt.show()






# %%


widgets.interact(exposure_simulation, n = widgets.Play(min = 20 , max = 800, step = 20, interval=200),baselineMTM = widgets.IntSlider(min=0,max=100000,step=500,value=1000,description="BaseMTM"),
         baselineNotional = widgets.IntSlider(min=0,max=5000,step=20,value=200,description="BaseNotional"),
         mtmMin = widgets.IntSlider(min=-20000,max=0,step=50,value=-2000,description="MTM Min"),
         mtmMax = widgets.IntSlider(min=0,max=20000,step=50,value=3000,description="MTM Max"),
         notionalMin = widgets.IntSlider(min=-2000,max=0,step=50,value=-1000,description="NotionalMin"),
         notionalMax = widgets.IntSlider(min=0,max=2000,step=50,value=1000,description="NotionalMax"),
         addOnWidget = widgets.FloatSlider(min=0,max=1,step=0.01,value=0.01,description="AddOnFactor"),
)


