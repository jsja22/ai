import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    

x = np.arange(-5,5,0.1)
y = relu(x)
print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()

####elu, selu, reaky_relu