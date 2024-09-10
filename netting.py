# if ISDA_A:
#   Add position x to risk exposure
# if ISDA_B:
#   Add position y to risk exposure
# if No Agreement:
#   Calculate risk individually
# 
#   Sum all risks together : rho(x1 + x2) + rho(x3 + x4) + rho(x5) + rho(x6) + rho(x7) + ....    <--- Brute Force approach

# Linear Approach : rho(baseline position) + sum(rho(baseline position + x1) - rho(baseline position))

import numpy as np
import random
import matplotlib.pyplot as plt


def PSR(positions, addonfactor, singleposition):

    MtM = 0
    AddOn = 0

    if singleposition == True:
        MtM += positions[0]
        AddOn += addonfactor * positions[1]
    else:
        for i in range(len(positions)):
            MtM += positions[i,0]
            AddOn += addonfactor * positions[i,1]

    CurrentExposure = max(0,MtM)

    PSR = CurrentExposure + AddOn
    
    return PSR

def PSR_BruteForce(positions, addonfactor, singleposition):

    netting_agreement_A = []
    netting_agreement_B = []
    netting_agreement_C = []
    netting_agreement_D = []
    netting_agreement_E = []


    for i in range(len(positions)):
        if positions[i,2] == 1:
            netting_agreement_A.append(positions[i,:])


def PSR_Conservative(positions, addonfactor):

    total_PSR = 0
    singleposition = True

    for i in range(len(positions)):
        total_PSR += PSR(positions[i,:], addonfactor, singleposition)
    
    return [total_PSR]

def PSR_Linear(positions, addonfactor):

    total_PSR = 0

    base_PSR = 0

    singleposition = True
    sample = positions[0,:]
    
    base_PSR += PSR(sample, addonfactor, singleposition)
    total_PSR = base_PSR
    singleposition = False

    for i in range(len(positions)-1):
        total_PSR += (PSR(positions[[0,i+1], :], addonfactor, singleposition) - base_PSR)


    return [total_PSR]


def PSR_Average(positions, addonfactor, n):

    total_PSR = 0
    base_PSR = 0

    singleposition = True
    base_PSR += PSR(positions[0,:], addonfactor, singleposition)
    total_PSR = base_PSR
    singleposition = False

    for i in range(len(positions)-1):

        # take out the baseline position and the x_i position
        trade_i = positions[[0,i+1], :]
        
        # multiply the i-th position by n - as it happens in the formula
        trade_i[1,:] *= n

        # multiply the formula by n
        total_PSR += 1/n*(PSR(trade_i, addonfactor, singleposition) - base_PSR)


    return [total_PSR]


def main(n):

    
    # Randomly generated set of [MTM, Notional]
   
    positions = np.zeros((n,2))

    positions[0,0] = 100
    positions[0,1] = 1000

    for i in range(len(positions)-1):
        positions[i+1,0] = random.randint(-10,10)
        positions[i+1,1] = abs(5*positions[i,0])


    addonfactor = 0.01
    singleposition = False


    print("\n")
    print("[PSR NGR, PSR Normal] for Brute Force: ", [PSR(positions, addonfactor, singleposition)])
    print("\n")

    print("[PSR NGR, PSR Normal] for Linearisation: ", PSR_Linear(positions, addonfactor))
    print("\n")
    print("[PSR NGR, PSR Normal] for Conservative: ", PSR_Conservative(positions, addonfactor))
    print("\n")
    print("[PSR NGR, PSR Normal] for Averages: ", PSR_Average(positions, addonfactor, 3))
    
    print("\n")
    print("Positions Matrix: ", positions)


main(20)


