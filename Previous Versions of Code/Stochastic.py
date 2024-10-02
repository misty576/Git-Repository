import numpy as np
import random
import matplotlib.pyplot as plt


addonfactor = 0.01

np.random.seed(2020)

X = np.random.uniform(-1000,1000,10**5)

Y = [max(0,i) + addonfactor * 10 * abs(i) for i in X]

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
    x[0] = 20000

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

