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
