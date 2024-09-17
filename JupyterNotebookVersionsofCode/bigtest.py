# %% [markdown]
# ## Strategies Investigation Notebook 1
# 
# #### So far this program investigates the potential exposure of each strategy via. Monte Carlo Simulation, and also investigates the percentage in growth from the brute force approach for each method!

# %% [markdown]
# ### SETUP: Define all the programs we will use

# %% [markdown]
# #### Step 1: Import Modules

# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import scipy
from scipy.stats import chisquare
import sys

# %% [markdown]
# #### Step 3: Define the 4 different strategies 

# %%
def PSR_NGR(positions, addonfactor, singleposition):

    GrossExposure = 0
    CurrentExposure = 0
    NetExposure = 0
    AddOn_Gross = 0

    if singleposition == True:
        GrossExposure += max(0, positions[0])
        CurrentExposure += positions[0]
        AddOn_Gross += addonfactor * positions[1]
    else:
        for i in range(len(positions)):
            GrossExposure += max(0, positions[i,0])
            CurrentExposure += positions[i,0]
            AddOn_Gross += addonfactor * positions[i,1]

    NetExposure = max(0, CurrentExposure)

    if GrossExposure == 0:
        NGR = 0
    else:
        NGR = NetExposure/GrossExposure
    
    PSR = max(0, CurrentExposure) + NGR * (0.6) * (AddOn_Gross) + (0.4) * (AddOn_Gross)

    return PSR



def PSR_Conservative(positions, addonfactor):

    total_PSR_NGR = 0
    singleposition = True

    for i in range(len(positions)):
        total_PSR_NGR += PSR_NGR(positions[i,:], addonfactor, singleposition)
    
    return total_PSR_NGR

def PSR_Linear(positions, addonfactor):

    total_PSR_NGR = 0

    base_PSR_NGR = 0
    singleposition = True
    sample = positions[0,:]
    base_PSR_NGR += PSR_NGR(sample, addonfactor, singleposition)
    total_PSR_NGR = base_PSR_NGR

    singleposition = False

    for i in range(len(positions)-1):
        total_PSR_NGR += (PSR_NGR(positions[[0,i+1], :], addonfactor, singleposition) - base_PSR_NGR)

    return total_PSR_NGR


def PSR_Average(positions, addonfactor, n):

    total_PSR_NGR = 0
    base_PSR_NGR = 0

    singleposition = True
    base_PSR_NGR += PSR_NGR(positions[0,:], addonfactor, singleposition)
    total_PSR_NGR = base_PSR_NGR

    singleposition = False

    for i in range(len(positions)-1):

        # take out the baseline position and the x_i position
        trade_i = positions[[0,i+1], :]
        
        # multiply the i-th position by n - as it happens in the formula
        trade_i[1,:] *= n

        # multiply the formula by n
        total_PSR_NGR += 1/n*(PSR_NGR(trade_i, addonfactor, singleposition) - base_PSR_NGR)


    return total_PSR_NGR

# %%
def main(n, baseline_mtm):


    # TEST 1: PSR_NGR = 30.765 , PSR_BruteForce = 30.9
    '''    
    positions = np.zeros((4,2))

    positions[0,0] = 10
    positions[1,0] = 20
    positions[2,0] = -10
    positions[3,0] = 10

    
    positions[0,1] = 20
    positions[1,1] = 40
    positions[2,1] = 10
    positions[3,1] = 20
    '''
    
    # Randomly generated set of [MTM, Notional]
   
    positions = np.zeros((n,2))

    positions[0,0] = baseline_mtm
    positions[0,1] = 1000

    for i in range(len(positions)-1):
        positions[i+1,0] = random.randint(-10,10)
        positions[i+1,1] = abs(5*positions[i,0])


    
    # TEST 1A
    '''
    positions = np.zeros((2,2))

    positions[0,0] = 10
    positions[1,0] = 20
    
    positions[0,1] = 20
    positions[1,1] = 40
    '''

    # TEST 2: PSR_NGR = 15.51 , PSR_BruteForce = 15.6

    '''
    positions = np.zeros((3,2))

    positions[0,0] = 10
    positions[1,0] = -5
    positions[2,0] = 10
    
    positions[0,1] = 20
    positions[1,1] = 10
    positions[2,1] = 30


    '''
    addonfactor = 0.01
    singleposition = False

    NGR_BF = PSR_NGR(positions, addonfactor, singleposition)

    NGR_Lin= PSR_Linear(positions, addonfactor)
    NGR_Cons = PSR_Conservative(positions, addonfactor)
    NGR_Avg = PSR_Average(positions, addonfactor, n)



    return [positions, addonfactor, NGR_BF, NGR_Lin, NGR_Cons, NGR_Avg]


[positions, addonfactor, NGR_BF, NGR_Lin, NGR_Cons, NGR_Avg] = main(20, 100)

print("\n")
print("[PSR NGR, PSR Normal] for Brute Force: ", NGR_BF)
print("\n")

print("[PSR NGR, PSR Normal] for Linearisation: ", PSR_Linear(positions, addonfactor))
print("\n")
print("[PSR NGR, PSR Normal] for Conservative: ", PSR_Conservative(positions, addonfactor))
print("\n")
print("[PSR NGR, PSR Normal] for Averages: ", PSR_Average(positions, addonfactor, 20))

print("\n")
print("Positions Matrix: \n", positions)


# %% [markdown]
# ### NB: Set up Baseline Position and Incoming Trades 

# %%
# BASELINE POSITION
MarkToMarket = 10000
Notional = 1000000


# For the incoming positions , we simply set the range of the values used for sampling incoming trades
MTM_Min = -10
MTM_Max = 10


# For simplicity, we set the AddOnFactor to be the same for all the trades
addOnFactor = 0.01

# %% [markdown]
# #### Step 5: Generate the dataset: Baseline Position + Incoming Trades

# %%
def get_position_impacts(n):
    
    baseline_position = [MarkToMarket, Notional]
    #baseline_position = [63000, 200]
    mtm_notional_matrix = np.zeros((n,2))

    
    for i in range(n):

        mtm_notional_matrix[i,0] = random.randint(MTM_Min,MTM_Max)
        mtm_notional_matrix[i,1] = abs(mtm_notional_matrix[i,0])*10
        
        #mtm_notional_matrix[i,0] = random.randint(-2000,3000)
        #mtm_notional_matrix[i,1] = random.randint(-1000,1000)
        
    positions = np.vstack([baseline_position, mtm_notional_matrix])
    return positions

# %% [markdown]
# #### Step 6: Calculate Mean and Variance of the respective methods by analysing the distribution of values at 100th trade

# %%
def get_mean_variance(final_vals, samples):
    
    mu = []
    sigma = []
    std = []

    for i in range(final_vals.shape[0]):
        mu.append(np.sum(final_vals[i,:])/samples)

        current_mu = np.sum(final_vals[i,:])/samples

        std.append(np.std(final_vals[i,:]))

        diff_squared = np.sum((final_vals[i,:] - current_mu) ** 2)
        sigma.append(diff_squared / samples)
        
    return [mu,sigma, std]

# %% [markdown]
# #### Step 6a: Clean data by rounding float numbers to 4 decimal places

# %%
def remove_decimal_places(my_list):
    new_list = list(np.around(np.array(my_list), 4))
    return new_list

# %% [markdown]
# ### PART 1: Monte Carlo Simulation of the strategies
# #### In our first simulation, we investigate the distribution of the exposure strategies.
# #### We generate a random dataset (baseline + positions) and input it into each exposure calculator. We repeat this process multiple times to get a distribution. 

# %% [markdown]
# #### Step 7: The big part: Simluate the exposure calculations using Monte Carlo

# %%
def exposure_simulation1(n, samples, addOnFactor, widget):

    x = np.linspace(0,n,n)
    final_vals = np.zeros((4,samples))

    fig,axs = plt.subplots(2,2, figsize=(10,8))
    fig.subplots_adjust(hspace=0.5) 
    fig.subplots_adjust(wspace=0.5) 

    singleposition = False

    for i in range(samples):
        
        bf_list = []
        lin_list = []
        cons_list = []
        avg_list = []

        positions = get_position_impacts(n)
        
        
        for j in range(len(positions)-1):
            
            #slice = positions[:j+1,:]

            slice = np.zeros((3,2))

            slice[0,0] = 10
            slice[1,0] = -5
            slice[2,0] = 10
            
            slice[0,1] = 20
            slice[1,1] = 10
            slice[2,1] = 30


            NGR_BF = PSR_NGR(slice, addOnFactor, singleposition)
            NGR_Lin = PSR_Linear(slice, addOnFactor)
            NGR_Cons = PSR_Conservative(slice, addOnFactor)
            NGR_Avg = PSR_Average(slice, addOnFactor, n)

            bf_list.append(NGR_BF)
            lin_list.append(NGR_Lin)
            cons_list.append(NGR_Cons)
            avg_list.append(NGR_Avg)
        
        axs[0,0].plot(x,bf_list, c='r', label="Brute Force")
        axs[0,1].plot(x,lin_list, 'g', label="Linearisation")
        axs[1,0].plot(x,cons_list, 'b', label="Conservative")
        axs[1,1].plot(x,avg_list, 'y', label="Averages")
        

        final_vals[0,i] = bf_list[-1]
        final_vals[1,i] = lin_list[-1]
        final_vals[2,i] = cons_list[-1]
        final_vals[3,i] = avg_list[-1]


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

    plt.show()
    
    if widget == False:
        return final_vals
    else:
        return None    


# %% [markdown]
# #### Step 8: Run the functions above, analyse the graphs, and inspect mean and variance of each method

# %%
samples = 500
n = 10
addonfactor = 0.01

final_vals = exposure_simulation1(n, samples, addOnFactor, False)

[mu,sigma,std] = get_mean_variance(final_vals, samples)


print("Entries below will correspond as follows : [Brute Force, Linearisation, Conservative, Averages]")
print("Average Values at trade 100: ", remove_decimal_places(mu))
print("Variance at trade 100: ", remove_decimal_places(sigma))
print("Standard Deviation at trade 100: ", remove_decimal_places(std))

# %% [markdown]
# #### INTERACT
# ##### Toggle the sliders below to investigate the growth rates yourself!

# %%
widgets.interact(exposure_simulation1, 
         n = widgets.IntSlider(min = 20 , max = 400, step = 1, value=100, description="No. of Trades"),
         samples = widgets.IntSlider(min=0,max=500,step=20,value=200,description="No. of Samples"),
         addOnFactor = widgets.FloatSlider(min=0, max = 1, value=0.01, step = 0.01, description="addOnFactor"),
         widget = True,
)

# %% [markdown]
# #### Step 8a: This is purely for fun, but we can show that the conservative approach follows an Arithmetic Brownian Motion (ABM). All we need to do is calculate the mean and standard deviation of Y = max(0,X) + 0.1*abs(X)

# %%

np.random.seed(2020)

X = np.random.uniform(MTM_Min,MTM_Max,10**5)

Y = [max(0,i) + addOnFactor * 10 * abs(i) for i in X]

# This monte carlo simluates the mean and variance, but maybe also show how it's done by hand as well !

mu = np.mean(Y)
standard_dev = np.std(Y)

dt = 1

print("Mean of distribution: " + str(mu) + "  Standard Deviation of distribution: " + str(standard_dev))
print("\n")
T = 100  # T = Time <=> Trade 

n = int(T/dt)
t = np.linspace(0,T,n)

def run_sim():

    x = np.zeros(n)
    x[0] = MarkToMarket + 0.01 * Notional

    for i in range(n-1):
        # This is our ABM equation. There are more stochastic calculus notes on the paper. 
        x[i+1] = x[i] + mu*dt + standard_dev*np.random.normal(0,1)*np.sqrt(dt)
    
    return x

plt.figure()
for i in range(500):
    x = run_sim()

    plt.plot(t, x, 'r')


print("mu = ", mu)
print("sigma = ", standard_dev)
mu_T = mu*T
sigma_T = standard_dev ** 2 * T
print("Distribution of trades at trade 100 --> N( mu * t + X_0 , sigma^2 * t ) = N( " + str(mu*T + 20000) + " , " + str(standard_dev ** 2 * T) + " )")
plt.xlabel("Trade number")
plt.ylabel("Exposure")
plt.show()



# %% [markdown]
# ### Part 2: Percentage Growth relative to Brute Force Method
# #### The alternative strategies by definition overestimate the exposure, but by how much ?
# #### The code below investigate the growth on an interactive graph, where you can see how the strategies work for different positions
# 

# %% [markdown]
# #### OPTIONAL: Perform a data clean to remove outliers in the percentage growth. 
# Often when the trdae number is small, the rate of growth can be exponential, so we can remove them using the function below 

# %%
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
    

# %% [markdown]
# #### Step 1: Run the code below to simluate percentage growth

# %%
def simulation_exposure_growth(n, addon, mtm, notional, MTM_Min, MTM_Max, lin=True, cons=False, avg=False, dataclean=False):


    bf_exposure_for_trade_i = []
    cons_exposure_for_trade_i = []
    lin_exposure_for_trade_i = []
    avg_exposure_for_trade_i = []

    for i in range(1,n):
        
        bf_list = []
        cons_list = []
        lin_list = []
        avg_list = []

        positions = get_position_impacts(i)
        
        singleposition = False

        NGR_BF = PSR_NGR(positions, addOnFactor, singleposition)
        NGR_Lin = PSR_Linear(positions, addOnFactor)
        NGR_Cons = PSR_Conservative(positions, addOnFactor)
        NGR_Avg = PSR_Average(positions, addOnFactor, n)


        bf_exposure_for_trade_i.append(NGR_BF)
        cons_exposure_for_trade_i.append(NGR_Cons)
        lin_exposure_for_trade_i.append(NGR_Lin)
        avg_exposure_for_trade_i.append(NGR_Avg)


    diff_cons = [(cons_exposure_for_trade_i[i]-bf_exposure_for_trade_i[i])/bf_exposure_for_trade_i[i] * 100 for i in range(len(bf_exposure_for_trade_i))]
    diff_lin = [(lin_exposure_for_trade_i[i]-bf_exposure_for_trade_i[i])/bf_exposure_for_trade_i[i] * 100 for i in range(len(bf_exposure_for_trade_i))]
    diff_avg = [(avg_exposure_for_trade_i[i]-bf_exposure_for_trade_i[i])/bf_exposure_for_trade_i[i] * 100 for i in range(len(bf_exposure_for_trade_i))]
        
    if dataclean == True:
        diff_cons = remove_outliers(diff_cons, 3)
        diff_lin = remove_outliers(diff_lin, 3)
        diff_avg = remove_outliers(diff_avg, 3)


    x = np.arange(1,n)

    plt.figure(figsize=(8, 5))
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

        

# %% [markdown]
# #### Run the code using the ipywidgets module

# %%
widgets.interact(simulation_exposure_growth, 
         n = widgets.Play(min = 20 , max = 800, step = 20, interval=200),
         addon = widgets.FloatSlider(min=0,max=1,step=0.01,value=0.01,description="AddOnFactor"),
         mtm = widgets.IntSlider(min=0,max=10000,step=100,value=10000,description="Base MTM"),
         notional = widgets.IntSlider(min=0,max=1000000,step=1000,value=100000,description="Base Notional"),
         MTM_Min = widgets.IntSlider(min=-10000,max=-100,step=50,value=-1000,description="MTM Min"),
         MTM_Max = widgets.IntSlider(min=100,max=10000,step=50,value=1000,description="MTM Max"), 
)


