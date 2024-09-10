import random
import numpy as np
import matplotlib.pyplot as plt

# each unit in time represents a new trade coming in

addonfactor = 0.01

Y = 0
mu = 0
standard_dev = 0

for i in range(1,100):
    X = np.random.uniform(-1000,1000)
    

    Y = max(0, X) + addonfactor * 10 * abs(X)
    mu += np.mean(Y)
    standard_dev += np.std(Y)

 # Note@ Research into how to calculate the mean of a max(0, Y), where Y is a random variable!

dt = 1


print("Mean of distribution: " + str(mu) + "  Standard Deviation of distribution: " + str(standard_dev))
print("\n")
T = 100  # T = Time <=> Trade 

n = int(T/dt)
t = np.linspace(0,T,n)

def run_sim():

    x = np.zeros(n)
    x[0] = 20000
    for i in range(n-1):

        #mtm = np.random.uniform(-1000,1000)
        #notional = abs(mtm*10)
        #exp = mtm + notional*(0.01)
        #x[i+1] = x[i] + 1/100*mu*dt + 1/1000*sigma*exp

        x[i+1] = x[i] + mu*dt + standard_dev*np.random.randn()*np.sqrt(dt)

    
    return x

plt.figure()
for i in range(500):
    x = run_sim()

    plt.plot(t, x, 'r')


plt.xlabel("Trade number")
plt.ylabel("Exposure")
plt.show()