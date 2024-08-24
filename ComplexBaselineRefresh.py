# Complex Baseline Refresh

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

# The brute force takes all the impacts, sums them together, and calculates the bulk exposure
def psrBruteForce(position, impacts, addOnFactor, bf_list):

    # The way it works: We input up a list called 'bf_list', and for each step, we sum all the impacts into a variable called 'total' and 
    # calculate the exposure. We then add that number to the list, and proceed to the next impact/trade.

    total = 0
    totalMtm = position[0]
    totalNotional = position[1]
    
    total += totalExposure(totalMtm, totalNotional, addOnFactor)
    
    for i in range(len(impacts)):

        totalMtm += impacts[i][0]
        totalNotional += impacts[i][1]

    total = totalExposure(totalMtm, totalNotional, addOnFactor)
    bf_list.append(total)
    
    return bf_list


def psrConservative(position, impacts, addOnFactor, cons_list):
    
    # The way it works: We input up a list called 'cons_list', and for each step, we calculate the exposure for each individual 
    # impact and add that to the total. We then add that number to the list, and proceed to the next impact/trade.

    total = 0
    total += totalExposure(position[0], position[1], addOnFactor)


    for i in range(len(impacts)):
        total += totalExposure(impacts[i,0], impacts[i,1], addOnFactor)
    
    cons_list.append(total)

    return cons_list

def psrLinearisation(position, impacts, addOnFactor, lin_list):

    # The way it works: We input up a list called 'lin_list'. We then calculate the baseline exposure and use that in each step of our calculation.
    # We then take the exposure for each impact and 'subtract' from baseline exposure (MXWiki can explain this better)

    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure

    for i in range(len(impacts)):
        total += totalExposure(position[0]+impacts[i,0], position[1]+impacts[i,1], addOnFactor) - position_exposure
        
        # We only want positive exposures added to our list since this will results in the linearisation approach producing incorrect answers
    lin_list.append(max(0, total))

        # lin_list.append(total)

    return lin_list

def psrAverages(position, impacts, addOnFactor, n, avg_list):

    # This method works similarly to the Linearisation approach.

    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure

    for i in range(len(impacts)):
        total += 1/n*(totalExposure(position[0]+n*impacts[i,0], position[1]+n*impacts[i,1], addOnFactor) - position_exposure)
    
    avg_list.append(total)

    return avg_list

def remove_outliers(data, threshold = 2):

    # The growth in exposure from the baseline can be exceptionally high for a small number of trades (since we are going
    # from 0% difference to, for instance, a 20% differnce depending on the impacts we use). This function removes the outliers 
    # that impact our analysis of the graph, by using standard mean and variance approaches.

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
    


def get_position_impacts(n):

    # This function generates the baseline position and the subsequent impacts.

    baseline_position = [400, 2000]
    mtm_notional_matrix = np.zeros((n,4))


    # Here we can set the range of values our impacts can take.
    for i in range(n):

        mtm_notional_matrix[i,0] = random.randint(-200,200)
        mtm_notional_matrix[i,1] = random.randint(0,250)
        mtm_notional_matrix[i,2] = random.randint(0,1)

    return [baseline_position, mtm_notional_matrix]



# the following function prevents the simluation below from regenerating a new dataset (baseline + impacts) for each iteration
position_impacts_state = {"n": 0, "data": None}
def update_position_impacts(n):
    global position_impacts_state
    if position_impacts_state["n"] != n:
        position_impacts_state["n"] = n
        position_impacts_state["data"] = get_position_impacts(n)


# Here we have our main function, which takes the parameters:
# n - number of impacts
# m - how often we refresh the baseline (for every 'm' trades)
# addOn - the addOn factor, which is set by the user

def exposure_simulation(n, addOn):
    
    update_position_impacts(n)
    baseline_position, mtm_notional_matrix = position_impacts_state["data"]
    
    addOnVal = addOn

    bf_list = []
    cons_list = []
    lin_list = []
    avg_list = []
    refresh_points = []

    validated_trades = np.zeros((1,4))
    validated_trades_cycle = np.zeros((1,4))
    failed_trades = np.zeros((1,4))
    validated_trades = np.delete(validated_trades, 0, axis=0)
    failed_trades = np.delete(failed_trades, 0, axis=0)
    validated_trades_cycle = np.delete(validated_trades_cycle, 0, axis=0)

    bf_list.append(totalExposure(baseline_position[0], baseline_position[1], addOn))
    cons_list.append(totalExposure(baseline_position[0], baseline_position[1], addOn))
    lin_list.append(totalExposure(baseline_position[0], baseline_position[1], addOn))
    avg_list.append(totalExposure(baseline_position[0], baseline_position[1], addOn))

    refresh_number = 0
    trade_number = 0

    while mtm_notional_matrix.shape[0] != 0:

        # We first assign the number of trades we shall need to validate before doing a refresh 

        if mtm_notional_matrix.shape[0] < 2:
            m = mtm_notional_matrix.shape[0]
        else:
            m = random.randint(1,2)
            if len(refresh_points) == 0:
                refresh_points.append(m)
            else:
                refresh_points.append(m + refresh_points[len(refresh_points)-1]+1)


        index = 0
        count = 0
        while m > 0:

            temp = mtm_notional_matrix[index,:]
            mtm_notional_matrix[index,2] = random.randint(0,2)

            if mtm_notional_matrix[index,2] < 2:

                # pass to exposure calculators
                validated_trades = np.vstack((validated_trades, temp))
                validated_trades_cycle = np.vstack((validated_trades_cycle, temp))
                print(validated_trades_cycle)
                [bf_list, cons_list, lin_list, avg_list] = [psrBruteForce(baseline_position, validated_trades_cycle, addOnVal, bf_list),
                                                            psrConservative(baseline_position, validated_trades_cycle, addOnVal, cons_list),
                                                            psrLinearisation(baseline_position, validated_trades_cycle, addOnVal, lin_list),
                                                            psrAverages(baseline_position, validated_trades_cycle, addOnVal, n, avg_list)]
        
                m -= 1

                # remove from list and add to temporary validated trades list

                mtm_notional_matrix = np.delete(mtm_notional_matrix, index, axis=0)
                trade_number += 1

            else:
                
                # move pointer to next item in list

                mtm_notional_matrix[index,3] += 1

                if mtm_notional_matrix[index,3] == 4:
                    # remove from list
                    failed_trades = np.vstack((failed_trades, temp))
                    mtm_notional_matrix = np.delete(mtm_notional_matrix, index, axis=0)
                
                index += 1
            count += 1
            if index == mtm_notional_matrix.shape[0]:
                
                # Go back to the original while loop
                break
        


        # We now perform the refresh. This can be done by taking in all the validated trades
        # and putting them in the baseline refresh mechanism
        mtmRefresh = baseline_position[0] + np.sum(validated_trades[:,0])
        notionalRefresh = baseline_position[1] + np.sum(validated_trades[:,1])
        baseline_position = [max(0, mtmRefresh), notionalRefresh]
        refresh_number += 1

        bf_list.append(totalExposure(max(0, mtmRefresh), notionalRefresh, addOn))
        cons_list.append(totalExposure(max(0, mtmRefresh), notionalRefresh, addOn))
        lin_list.append(totalExposure(max(0, mtmRefresh), notionalRefresh, addOn))
        avg_list.append(totalExposure(max(0, mtmRefresh), notionalRefresh, addOn))

        validated_trades_cycle = np.zeros((1,4))
        validated_trades_cycle = np.delete(validated_trades_cycle, 0, axis=0)
        
        if mtm_notional_matrix.shape[0] == 0:
            refresh_points.append(m + refresh_points[len(refresh_points)-1]+1)

                    


    ####################    POST-PROCESSING    ####################
    

    # Here we plot our analyses
    
    x = np.linspace(0, trade_number + refresh_number, trade_number + refresh_number+1)

    print(bf_list)
    print(trade_number)
    print(refresh_number)
    plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(x, bf_list, 'b', label = "BF")
    plt.plot(x, cons_list, 'r', label = "Conservative")
    plt.plot(x, lin_list, 'g', label = "Linearisation")
    plt.plot(x, avg_list, 'y', label = "Averages")
    plt.xlabel("Trade Number")
    plt.ylabel("Exposure")

    print(refresh_points)
    for i in range(len(refresh_points)):

        vline_x = refresh_points[i]+1
        ymin, ymax = plt.ylim()
        plt.axvline(vline_x, color='k', linestyle='--')
        vline_x2 = refresh_points[i]
        ymin2, ymax2 = plt.ylim()
        plt.axvline(vline_x2, color='r', linestyle='--')
        plt.text(vline_x, ymax + (ymax - ymin)*0.025, 'RF', ha="center", va = "bottom", color="k")
        plt.text(vline_x2, ymax2 + (ymax2 - ymin2)*0.025, '↻', ha="center", va = "bottom", color="r")

    vline_start = 0
    ymin3, ymax3 = plt.ylim()
    plt.axvline(vline_start, color='b', linestyle='--')
    plt.text(vline_start, ymax3 + (ymax3 - ymin3)*0.025, 'Base', ha="center", va = "bottom", color="b")

    plt.legend()
    plt.show()

    print("Impacts Matrix \n", mtm_notional_matrix)
    
    print("\n")
    
    print("BF list ", bf_list)
    print("Lin List", lin_list)
    
    print("\n")
    
    print("Conservative List", cons_list)
    print("Average List", avg_list)


exposure_simulation(4, 0.01)



# Issue: After a baseline refresh is done, the code is still using all the validated trades from the 
# previous baseline refresh in the exposure calculations. So what can be done next? There has to be 
# some way we track the trades per baseline refresh and only use the new ones.
# SOLVED THE ABOVE

# NEXT: During the second baseline refresh, why is the baseline moving upwards? This should not be happening