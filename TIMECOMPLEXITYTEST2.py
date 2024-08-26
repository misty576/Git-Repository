import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import time
import math
from IPython.display import clear_output

# New approach for calculating exposure

# position --> p + x1 + x2 + ... + xn

# exposure --> rho(p + x1 + x2 + ... + xn)


# what is the right way to calculate this exposure?

# 1. we store the position as such: position = [p, x1, x2, ... , xn] ,  where p, xi = [mtm, notional].
# 2. we set mtm = 0, notional = 0, and we create a for loop that does the following:
#       for i in position:
#           mtm += position[i][0]
#   	    notional += position[i][1]
#
# 2a. If we have only one position (just p for instance) then we skip the for loop.
#
# 3. We then take max(0,mtm) and notional*addon, add them together and we get our final result.


#def currentExposure(mtm):
#    return max(0,mtm)


#def addOnCalculator(notional, addon):
#    return notional*addon

global conservative
conservative = False


# New, more computationally expensive exposure calculation
def currentExposure(mtm):
    # Adding a more complex operation
    return max(0, mtm)

def addOnCalculator(notional, addon):
    # Adding a more complex operation
    return notional * addon


def totalExposure(position, addon, conservative):
    
    start_time = time.time()
    length = len(position)
    pos = type(position[0])
    if conservative == False:
        if len(position) == 1:
            currExp = currentExposure(position[0][0])
            currAddon = addOnCalculator(position[0][1], addon)
            check_policy(position)
            end_time = time.time()
            total_time = end_time - start_time
            
            return [currExp + currAddon, total_time]
        else:   
            mtm = 0
            notional = 0
            for pos in position:
                mtm += pos[0]
                notional += pos[1]
            currExp = currentExposure(mtm)
            currAddon = addOnCalculator(notional, addon)
            check_policy(position)
    else:
        # this is for the conservative approach
        currExp = currentExposure(position[0])
        currAddon = addOnCalculator(position[1], addon)
        check_policy(position)

        end_time = time.time()
        total_time = end_time - start_time
        
        return [currExp + currAddon, total_time]
    
    end_time = time.time()
    total_time = end_time - start_time
        
    return [currExp + currAddon, total_time]

# for some reason this code is passing into the else part of the statement when the length is still 1. why????


def check_policy(position):
    # this code simulates the time taken to check policy, and takes longer depending on how big your position becomes (no. of impacts in code)
    size_factor = len(position)    
    time.sleep(1e-10 * (size_factor**2))
    

    

def psrBruteForce(baseline, impacts, addon):
    
    np.append(baseline, impacts)
    position = baseline
    [total, total_time] = totalExposure(position, addon, conservative)


    return [total, total_time]
# NB: i DON'T THINK THIS CODE IS WORKING CORRECTLY, FOR SOME REASON IT IS ADDING THE BASELINE AND POSITION TOGETHER WTF. USE APPEND INSTEAD

def psrLinearisation(baseline, impacts, addon):

    total_time = 0
    exp_baseline = totalExposure(baseline, addon, conservative)
    check_policy(baseline)

    total = exp_baseline
    for i in range(len(impacts)):
        baseline_temp = baseline
        np.append(baseline_temp, impacts[i])
        [total_exp, time_exp] = totalExposure(baseline_temp, addon, conservative)
        total += total_exp - exp_baseline
        total_time += time_exp
        check_policy(baseline_temp)


    return [total, total_time]

def psrConservative(baseline, impacts, addon):
    
    np.append(baseline, impacts)
    position = baseline
    total = 0
    total_time = 0
    conservative = True
    for i in range(len(position)):
        [total_exp, time_exp] = totalExposure(position[i], addon, conservative)
        total += total_exp
        total_time += total_time
        check_policy(position[i])
    

    conservative = False
    return [total, total_time]

def psrAverages(baseline, impacts, addon, n):

    exp_baseline = totalExposure(baseline, addon, conservative)
    check_policy(baseline)

    total = exp_baseline
    total_time = 0

    for i in range(len(impacts)):
        baseline_temp = baseline
        np.append(baseline_temp, impacts[i])
        [total_exp, time_exp] = totalExposure(baseline_temp, addon, conservative)
        total_time += time_exp
        total += 1/n*(total_exp - exp_baseline)
        check_policy(baseline_temp)
    
    return [total, total_time]


def exposure_simulation(n, addon):

    # here we want to obtain a graph showing the growth of the difference between baseline/brute force as number of trades increase
    addOnVal = 0.01
    baseline = np.array([[10000, 1000000]])  


    bf_list = []
    cons_list = []
    lin_list = []
    avg_list = []

    times_bf = []
    times_cons = []
    times_lin = []
    times_avg = []
    fig,axs = plt.subplots(2,2, figsize=(10,8))

    for i in range(1,n):
    
        impacts = np.zeros((i,2)) # Column 1 will be MTM, Column 2 will be Notional

        for j in range(i):
            impacts[j,0] = random.randint(-2000,2000)
            impacts[j,1] = random.randint(0, 100000)


        [total_bf, time_bf] = psrBruteForce(baseline, impacts, addOnVal)
        [total_lin, time_lin] = psrLinearisation(baseline, impacts, addOnVal)
        [total_cons, time_cons] = psrConservative(baseline, impacts, addOnVal)
        [total_avg, time_avg] = psrAverages(baseline, impacts, addOnVal, n)

        bf_list.append(total_bf)
        cons_list.append(total_cons)
        lin_list.append(total_lin)
        avg_list.append(total_avg)

        times_bf.append(time_bf)
        times_cons.append(time_cons)
        times_lin.append(time_lin)
        times_avg.append(time_avg)

        clear_output(wait=True)
    
    x = np.arange(1,n)

    axs[0,0].plot(x,times_bf, c='r', label="Brute Force")
    axs[0,1].plot(x,times_lin, 'g', label="Linearisation")
    axs[1,0].plot(x,times_cons, 'b', label="Conservative")
    axs[1,1].plot(x,times_avg, 'y', label="Averages")


    axs[0,0].set_xlabel("No. of trades")
    axs[0,0].set_ylabel("Time to compute")

    axs[1,0].set_xlabel("No. of trades")
    axs[1,0].set_ylabel("Time to compute")

    axs[0,1].set_xlabel("No. of trades")
    axs[0,1].set_ylabel("Time to compute")

    axs[1,1].set_xlabel("No. of trades")
    axs[1,1].set_ylabel("Time to compute")
    
    axs[0,0].set_title("Brute Force")
    axs[0,1].set_title("Linearisation")
    axs[1,0].set_title("Conservative")
    axs[1,1].set_title("Averages")
    plt.show()


exposure_simulation(100, 0.01)
