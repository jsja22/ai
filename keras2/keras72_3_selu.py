import numpy as np
import matplotlib.pyplot as plt

def selu(x, a =1.67326, l =1.0507):
    y_list = []
    for x in x:
        if x >= 0:
            y = l * x
        if x < 0:
            y = l * a * (np.exp(x) - 1)
        y_list.append(y)
    return y_list


x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()
