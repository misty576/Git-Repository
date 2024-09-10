import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2020)

x = np.random.uniform(-100,100,10**5)

y = np.abs(x)

plt.figure()
plt.subplot(1,2,1)

plt.hist(x, bins=12, density=True, color="skyblue", edgecolor="black")
plt.title("histogram of x")
plt.xlabel("x values")
plt.ylabel("Density")

plt.subplot(1,2,2)

plt.hist(y, bins=12, density=True, color="skyblue", edgecolor="black")
plt.title("histogram of y")
plt.xlabel("y values")
plt.ylabel("Density")

plt.tight_layout()
plt.show()


mean1 = np.mean(x)
mean2 = np.mean(y)

std1 = np.std(x)
std2 = np.std(y)

print("Mean of X: " + str(mean1) + "  Standard Deviation of X: " + str(std1))
print("\n")
print("Mean of Y: " + str(mean2) + "  Standard Deviation of Y: " + str(std2))