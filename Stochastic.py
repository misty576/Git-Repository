import random
import numpy as np
import matplotlib.pyplot as plt

# each unit in time represents a new trade coming in

addonfactor = 0.01
X = np.random.uniform(0,1000, 10**6)
Z = 10 * abs(X)
mu_test = np.mean(X)
sigma_test = np.std(X)
X_rv = X + addonfactor*Z
mu = np.mean(X_rv)
sigma = np.std(X_rv)

dt = 1


print("Mean of distribution: " + str(mu) + "  Standard Deviation of distribution: " + str(sigma))
print("\n")
T = 100  # T = Time <=> Trade 

n = int(T/dt)
t = np.linspace(0,T,n)

def run_sim():
    x = np.zeros(n)
    x[0] = 20000
    for i in range(n-1):
        mtm = np.random.uniform(-1000,1000)
        notional = abs(mtm*10)
        exp = mtm + notional*(0.01)
        #x[i+1] = x[i] + 1/100*mu*dt + 1/1000*sigma*exp
        x[i+1] = x[i] + mu*dt + sigma*np.random.randn()*np.sqrt(dt)

    
    return x

plt.figure()
for i in range(500):
    x = run_sim()

    plt.plot(t, x, 'r')


plt.xlabel("Trade number")
plt.ylabel("Exposure")
plt.show()