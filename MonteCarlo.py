# In this code we perform Monte Carlo simulation on the different methods

# What we are going to do is take a baseline position [mtm, notional] with a set Addon and we are going to generate 1000 different data sets where each dataset will be a bunch of trades with [mtm, notional]

# What we will do is take a plot of how to the exposure changes over time for each dataset when we pick a particular method, so this way we should end up with some sort of distribution for the strategies

import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import scipy
from scipy.stats import chisquare

# new comment added here

def currentExposure(mtm):
    return max(0,mtm)


def addOn(notional, addOnFactor):
    return notional*addOnFactor


def totalExposure(mtm, notional, addOnFactor):
    currExp = currentExposure(mtm)
    currAddOn = addOn(notional, addOnFactor)

    return currExp + currAddOn

def psrBruteForce(position, impacts, addOnFactor, bf_list):
    
    total = 0
    totalMtm = position[0]
    totalNotional = position[1]
    
    total += totalExposure(totalMtm, totalNotional, addOnFactor)
    bf_list.append(total)
    
    for i in range(len(impacts)):

        totalMtm += impacts[i,0]
        totalNotional += impacts[i,1]
        total = totalExposure(totalMtm, totalNotional, addOnFactor)
        bf_list.append(total)
    
    return bf_list


def psrConservative(position, impacts, addOnFactor, cons_list):
    total = 0
    total += totalExposure(position[0], position[1], addOnFactor)
    cons_list.append(total)

    for i in range(len(impacts)):
        total += totalExposure(impacts[i,0], impacts[i,1], addOnFactor)
        cons_list.append(total)

    return cons_list

def psrLinearisation(position, impacts, addOnFactor, lin_list):
    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure
    lin_list.append(position_exposure)

    for i in range(len(impacts)):
        total += totalExposure(position[0]+impacts[i,0], position[1]+impacts[i,1], addOnFactor) - position_exposure
        
        # We implement the workaround as dicussed below
        lin_list.append(max(0, total))

        # lin_list.append(total)

    return lin_list

def psrAverages(position, impacts, addOnFactor, n, avg_list):
    total = 0
    position_exposure = totalExposure(position[0], position[1], addOnFactor)
    total += position_exposure
    avg_list.append(position_exposure)

    for i in range(len(impacts)):
        total += 1/n*(totalExposure(position[0]+n*impacts[i,0], position[1]+n*impacts[i,1], addOnFactor) - position_exposure)
        avg_list.append(total)

    return avg_list




def get_position_impacts(n):
    
    baseline_position = [10000000, 1000000]
    mtm_notional_matrix = np.zeros((n,2))

    
    for i in range(n):

        mtm_notional_matrix[i,0] = random.randint(-50000,50000)
        mtm_notional_matrix[i,1] = random.randint(0,250000)

    return [baseline_position, mtm_notional_matrix]

position_impacts_state = {"n": 0, "data": None}

def update_position_impacts(n):
    global position_impacts_state
    if position_impacts_state["n"] != n:
        position_impacts_state["n"] = n
        position_impacts_state["data"] = get_position_impacts(n)

def get_mean_variance(final_vals, samples):
    
    mu = []
    sigma = []
    sigma2 = []

    for i in range(final_vals.shape[0]):
        mu.append(np.sum(final_vals[i,:])/samples)

        sigma2.append(np.std(final_vals[i,:]))

        for j in range(final_vals.shape[0]):

            square = (final_vals[i,j] - mu[i]) ** 2

        sigma.append(np.sqrt(square/samples))

    return [mu,sigma, sigma2]



def exposure_simluation1(n, samples, addon):


    x = np.linspace(0,n+1,n+1)
    final_vals = np.zeros((4,samples))

    fig,axs = plt.subplots(2,2, figsize=(10,8))
    cons_list_backup = []
    for i in range(samples):
        bf_list = []
        lin_list = []
        cons_list = []
        avg_list = []


        [bl, matrix] = get_position_impacts(n)

        bf_result = psrBruteForce(bl, matrix, addon, bf_list)
        lin_result = psrLinearisation(bl, matrix, addon, lin_list)
        cons_result = psrConservative(bl, matrix, addon, cons_list)
        cons_list_backup.append(cons_result)
        avg_result = psrAverages(bl, matrix, addon, n, avg_list)
        
        axs[0,0].plot(x,bf_result, c='r', label="Brute Force")
        axs[0,1].plot(x,lin_result, 'g', label="Linearisation")
        axs[1,0].plot(x,cons_result, 'b', label="Conservative")
        axs[1,1].plot(x,avg_result, 'y', label="Averages")
        

        final_vals[0,i] = bf_list[-1]
        final_vals[1,i] = lin_list[-1]
        final_vals[2,i] = cons_list[-1]
        final_vals[3,i] = avg_list[-1]


    # this isn ot working right, final_vals is mostly a zeros matrix, and as a result, the standard deviation is just a multiple of the mean, how do I fix this?
    [mu,sigma,sigma2] = get_mean_variance(final_vals, samples)

    axs[0,0].set_xlabel("Trade Number")
    axs[0,0].set_ylabel("Exposure")

    axs[1,0].set_xlabel("Trade Number")
    axs[1,0].set_ylabel("Exposure")

    axs[0,1].set_xlabel("Trade Number")
    axs[0,1].set_ylabel("Exposure")

    axs[1,1].set_xlabel("Trade Number")
    axs[1,1].set_ylabel("Exposure")
    
    axs[0,0].set_title("Brute Force")
    axs[0,1].set_title("Linearisation")
    axs[1,0].set_title("Conservative")
    axs[1,1].set_title("Averages")


    print(final_vals)
    print(mu)
    print(sigma)
    plt.show()


    plt.figure()
    

    for i in range(samples):

        plt.plot(x,cons_list_backup[i] - 1/1000*mu[2]*x, 'b', label="Conservative")

    plt.title("Conservative Approach (after Shear Trasformation)")
    plt.xlabel("Trade Number")
    plt.ylabel("Exposure (Not Scaled)")
    plt.show()

    for i, method in enumerate(['Brute Force', 'Linearisation', 'Conservation', 'Averages']):
        observed_data = final_vals[i,:]

        expected_data = np.random.normal(loc = mu[i], scale = sigma[i], size=samples)
        observed_data_normalized = observed_data / observed_data.sum()
        expected_data_normalized = expected_data / expected_data.sum()

        #print(np.sum(observed_data))
        #print(np.sum(expected_data))
        chi2_stat, p_val = chisquare(observed_data, np.sum(observed_data)/np.sum(expected_data) * expected_data)
        #chi2_stat, p_val = chisquare(observed_data_normalized, expected_data_normalized)


        print(f"{method} Method: ")
        print(f"Chi2 Statistic: {chi2_stat} ")
        print(f"P-value: {p_val}\n ")




exposure_simluation1(250, 250, 0.01)


# If we set the baseline mtm to be really low, we get an interesting result. For the brute force graph, we observe that we get a linear curve going up at the bottom, which I suspect is the effect of the Notional * AddOn factor coming into play


# I want to do a shear transformation on the conservative graph, to convert it to a normal distribution again, try and figure out a way to apply a shear transformation on the graph, since right now itis plotting each one as it ais calculated, and then it calculates the mean and variance afterwards, which is annoying.