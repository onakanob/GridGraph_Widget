# -*- coding: utf-8 -*-
"""
Testing autograd
"""

import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
from matplotlib import pyplot as plt

def myfun(x):
    return np.multiply(x, 2)

def tanh(x):
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)
    
mygrad = grad(myfun)

print(grad(myfun)(3.))  # 2

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),                                     # first  derivative
         x, egrad(egrad(tanh))(x),                              # second derivative
         x, egrad(egrad(egrad(tanh)))(x))                       # third  derivative

plt.show()