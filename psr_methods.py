from .exposure_calculators import totalExposure

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
