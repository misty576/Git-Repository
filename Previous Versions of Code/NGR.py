# Gross MV = max(0, MtM)

# Net MV = MtM

# CurrentExposure = Current Exposure = Net MV = sum(mtm1 + mtm2 + ...)
# Gross Exposure = sum(max(0,mtm1) + max(0,mtm2) + ...) 
# Net Exposure = max(sum(mtm1 + mtm2 + ...), 0)
# NGR  = (Net Exposure) / (Gross Exposure)


# PSR = (Net MV) + NGR*(0.6)*(AddOn_Gross) + (0.4)*(AddOn_Gross)

# PSR = MtM + AddOn ---> AddOn = AddOn*addonfactor

# Example: p = [10, 20]  ,  x1 = [-5, 10]  ,  x2 = [10, 30]  ,  Addonfactor = 0.01
# CurrentExposure = 10 - 5 + 10 = 15
# NGR = GE/NE  -->  GE = 10 + 0 + 10 = 20  ,  NE = max(10 - 5 + 10, 0) = 15   --->   NGR = 15/20 = 3/4
# AddOn_Gross = 0.01 * 20 + 0.01 * 10 + 0.01 * 30 = 0.6

# PSR = 15 + (0.75 * 0.6 * 0.6) + (0.4 * 0.6) = 15.51
# Normal PSR = 15 + 0.01 * (20 + 10 + 30) = 15.6   ---> PSR is reduced by NGR method (incorporating netting)

import numpy as np
import random
import matplotlib.pyplot as plt

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

def PSR_Normal(positions, addonfactor, singleposition):

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


def PSR_Conservative(positions, addonfactor):

    total_PSR_NGR = 0
    total_PSR_Normal = 0
    singleposition = True

    for i in range(len(positions)):
        total_PSR_NGR += PSR_NGR(positions[i,:], addonfactor, singleposition)
        total_PSR_Normal += PSR_Normal(positions[i,:], addonfactor, singleposition)
    
    return [total_PSR_NGR, total_PSR_Normal]

def PSR_Linear(positions, addonfactor):

    total_PSR_NGR = 0
    total_PSR_Normal = 0

    base_PSR_NGR = 0
    base_PSR_Normal = 0

    singleposition = True
    sample = positions[0,:]
    base_PSR_NGR += PSR_NGR(sample, addonfactor, singleposition)
    total_PSR_NGR = base_PSR_NGR

    base_PSR_Normal += PSR_Normal(sample, addonfactor, singleposition)
    total_PSR_Normal = base_PSR_Normal
    singleposition = False

    for i in range(len(positions)-1):
        total_PSR_NGR += (PSR_NGR(positions[[0,i+1], :], addonfactor, singleposition) - base_PSR_NGR)
        total_PSR_Normal += (PSR_Normal(positions[[0,i+1], :], addonfactor, singleposition) - base_PSR_Normal)


    return [total_PSR_NGR, total_PSR_Normal]


def PSR_Average(positions, addonfactor, n):

    total_PSR_NGR = 0
    total_PSR_Normal = 0
    base_PSR_NGR = 0
    base_PSR_Normal = 0

    singleposition = True
    base_PSR_NGR += PSR_NGR(positions[0,:], addonfactor, singleposition)
    total_PSR_NGR = base_PSR_NGR

    base_PSR_Normal += PSR_Normal(positions[0,:], addonfactor, singleposition)
    total_PSR_Normal = base_PSR_Normal
    singleposition = False

    for i in range(len(positions)-1):

        # take out the baseline position and the x_i position
        trade_i = positions[[0,i+1], :]
        
        # multiply the i-th position by n - as it happens in the formula
        trade_i[1,:] *= n

        # multiply the formula by n
        total_PSR_NGR += 1/n*(PSR_NGR(trade_i, addonfactor, singleposition) - base_PSR_NGR)
        total_PSR_Normal += 1/n*(PSR_Normal(trade_i, addonfactor, singleposition) - base_PSR_Normal)


    return [total_PSR_NGR, total_PSR_Normal]



def main(n):


    # TEST 1: PSR_NGR = 30.765 , PSR_BruteForce = 30.9
    '''
    positions = np.zeros((3,2))

    positions[0,0] = 10
    positions[1,0] = 20
    positions[2,0] = -10
    positions[3,0] = 10

    
    positions[0,1] = 20
    positions[1,1] = 40
    positions[2,1] = 10
    positions[3,1] = 20
    '''
    '''
    positions = np.zeros((2,2))

    positions[0,0] = 10
    positions[1,0] = -5

    positions[0,1] = 20
    positions[1,1] = 10
    '''
    
    positions = np.zeros((2,2))

    positions[0,0] = 5
    positions[1,0] = 10
    
    positions[0,1] = 30
    positions[1,1] = 30
    

    # Randomly generated set of [MTM, Notional]
    '''   
    positions = np.zeros((n,2))

    positions[0,0] = 100
    positions[0,1] = 1000

    for i in range(len(positions)-1):
        positions[i+1,0] = random.randint(-10,10)
        positions[i+1,1] = abs(5*positions[i,0])

    '''

    
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


    print("\n")
    print("[PSR NGR, PSR Normal] for Brute Force: ", [PSR_NGR(positions, addonfactor, singleposition), PSR_Normal(positions, addonfactor, singleposition)])
    print("\n")

    print("[PSR NGR, PSR Normal] for Linearisation: ", PSR_Linear(positions, addonfactor))
    print("\n")
    print("[PSR NGR, PSR Normal] for Conservative: ", PSR_Conservative(positions, addonfactor))
    print("\n")
    print("[PSR NGR, PSR Normal] for Averages: ", PSR_Average(positions, addonfactor, 1))
    
    print("\n")
    #print("Positions Matrix: \n", positions)


main(50)


# Okay, so it seems that the methods are now working well.

# WARNING: I still need to run some basic tests on the linearisation, averages and conservative code to confirm that the code is still calculating correctly!!!!

