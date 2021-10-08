import numpy as np
import matplotlib.pyplot as plt

cooling_fun = lambda k: lambda_target + (lambda_init - lambda_target) * (1 / (1 + np.exp(((-13000 * np.log((np.abs(lambda_init - lambda_target))) / n) * ((k - n/2))))))
lambda_init = 1e-3
lambda_target = 1
lambda_ = lambda_init

n = 700
x = np.linspace(1, n, n)

fig = plt.figure()
plt.plot(x, cooling_fun(x))
plt.xlabel('$n$ cycles')
plt.ylabel('$\lambda$')
plt.grid()
plt.title('Exponential additive cooling function')
plt.show()
