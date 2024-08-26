import numpy as np
import random

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
