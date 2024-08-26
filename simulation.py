import numpy as np
import random
import matplotlib.pyplot as plt
from .exposure_calculators import totalExposure
from .psr_methods import psrBruteForce, psrConservative, psrLinearisation, psrAverages
from .utils import get_position_impacts
from .plotting import plot_exposure_simulation


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

        plot_exposure_simulation(trade_number, refresh_number, bf_list, cons_list, lin_list, avg_list, refresh_points)    

# Issue: After a baseline refresh is done, the code is still using all the validated trades from the 
# previous baseline refresh in the exposure calculations. So what can be done next? There has to be 
# some way we track the trades per baseline refresh and only use the new ones.
# SOLVED THE ABOVE

# NEXT: During the second baseline refresh, why is the baseline moving upwards? This should not be happening