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

def check_policy(position):
    # this code simulates the time taken to check policy, and takes longer depending on how big your position becomes (no. of impacts in code)
    size_factor = len(position)    
    #time.sleep(1e-10 * (size_factor**2))
    

# New, more computationally expensive exposure calculation
def currentExposure(mtm):
    # Adding a more complex operation
    return max(0, mtm)

def addOnCalculator(notional, addon):
    # Adding a more complex operation
    return notional * addon



def net_positions(position):
    # Grouping positions based on some criteria, for example instrument type (here simplified as 'instrument_id')
    # For now, assume all positions are of the same instrument type, so net everything together
    net_mtm = 0
    net_notional = 0
    
    for pos in position:
        net_mtm += pos[0]
        net_notional += pos[1]
    
    return [net_mtm, net_notional]

def totalExposureWithNetting(position, addon, conservative):
    start_time = time.time()
    
    # Net the positions first
    net_mtm, net_notional = net_positions(position)
    
    if conservative == False:
        currExp = currentExposure(net_mtm)
        currAddon = addOnCalculator(net_notional, addon)
    else:
        # Conservative calculation uses the first position directly
        currExp = currentExposure(position[0][0])
        currAddon = addOnCalculator(position[0][1], addon)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return [currExp + currAddon, total_time]

# Modify the other PSR calculation methods to use netting
def psrBruteForceWithNetting(baseline, impacts, addon):
    position = np.append(baseline, impacts, axis=0)
    [total, total_time] = totalExposureWithNetting(position, addon, conservative)
    return [total, total_time]

def psrLinearisationWithNetting(baseline, impacts, addon):
    exp_baseline = totalExposureWithNetting(baseline, addon, conservative)
    check_policy(baseline)

    total = exp_baseline[0]
    total_time = 0

    for impact in impacts:
        baseline_temp = np.append(baseline, [impact], axis=0)
        [total_exp, time_exp] = totalExposureWithNetting(baseline_temp, addon, conservative)
        total += total_exp - exp_baseline[0]
        total_time += time_exp
        check_policy(baseline_temp)

    return [total, total_time]

def psrConservativeWithNetting(baseline, impacts, addon):
    position = np.append(baseline, impacts, axis=0)
    total = 0
    total_time = 0
    conservative = True
    
    for pos in position:
        [total_exp, time_exp] = totalExposureWithNetting([pos], addon, conservative)
        total += total_exp
        total_time += time_exp
        check_policy([pos])

    conservative = False
    return [total, total_time]

def psrAveragesWithNetting(baseline, impacts, addon, n):
    exp_baseline = totalExposureWithNetting(baseline, addon, conservative)
    check_policy(baseline)

    total = exp_baseline[0]
    total_time = 0

    for impact in impacts:
        baseline_temp = np.append(baseline, [impact], axis=0)
        [total_exp, time_exp] = totalExposureWithNetting(baseline_temp, addon, conservative)
        total += 1/n*(total_exp - exp_baseline[0])
        total_time += time_exp
        check_policy(baseline_temp)
    
    return [total, total_time]

# Update the exposure simulation function to use netting
def exposure_simulation_with_netting(n, addon):
    baseline = np.array([[10000, 1000000]])  
    bf_list = []
    cons_list = []
    lin_list = []
    avg_list = []

    times_bf = []
    times_cons = []
    times_lin = []
    times_avg = []
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i in range(1, n):
        impacts = np.zeros((i, 2))  # Column 1 will be MTM, Column 2 will be Notional

        for j in range(i):
            impacts[j, 0] = random.randint(-2000, 2000)
            impacts[j, 1] = random.randint(0, 100000)

        [total_bf, time_bf] = psrBruteForceWithNetting(baseline, impacts, addon)
        [total_lin, time_lin] = psrLinearisationWithNetting(baseline, impacts, addon)
        [total_cons, time_cons] = psrConservativeWithNetting(baseline, impacts, addon)
        [total_avg, time_avg] = psrAveragesWithNetting(baseline, impacts, addon, n)

        bf_list.append(total_bf)
        cons_list.append(total_cons)
        lin_list.append(total_lin)
        avg_list.append(total_avg)

        times_bf.append(time_bf)
        times_cons.append(time_cons)
        times_lin.append(time_lin)
        times_avg.append(time_avg)

        clear_output(wait=True)

    x = np.arange(1, n)

    axs[0, 0].plot(x, times_bf, c='r', label="Brute Force with Netting")
    axs[0, 1].plot(x, times_lin, 'g', label="Linearisation with Netting")
    axs[1, 0].plot(x, times_cons, 'b', label="Conservative with Netting")
    axs[1, 1].plot(x, times_avg, 'y', label="Averages with Netting")

    axs[0, 0].set_xlabel("No. of trades")
    axs[0, 0].set_ylabel("Time to compute")

    axs[1, 0].set_xlabel("No. of trades")
    axs[1, 0].set_ylabel("Time to compute")

    axs[0, 1].set_xlabel("No. of trades")
    axs[0, 1].set_ylabel("Time to compute")

    axs[1, 1].set_xlabel("No. of trades")
    axs[1, 1].set_ylabel("Time to compute")

    axs[0, 0].set_title("Brute Force with Netting")
    axs[0, 1].set_title("Linearisation with Netting")
    axs[1, 0].set_title("Conservative with Netting")
    axs[1, 1].set_title("Averages with Netting")

    plt.show()

exposure_simulation_with_netting(2000, 0.01)
