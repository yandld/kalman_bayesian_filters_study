import numpy as np

x = np.array([1,2,3])
print(x)
x = np.array((4,5,6))
print(x)

x = np.array([[1,2,3],[4,5,6]])
print(x)

x = np.array([1,2,3], dtype=float)
x = np.array([[1,2,3],[4,5,6]])


## numpy matrix operation
x = np.array([[1., 2.],[3., 4.]])
print('x', x)
print('addition:\n', x + x)
print('\nelment-wise multiplication\n', x * x)
print('\nmultiplication\n', np.dot(x, x))
print('\ndot is also a member of np.array\n', x.dot(x))

import scipy.linalg as linalg

print('tanspose\n', x.T)
print('Numpy ninverse\n', np.linalg.inv(x))
print('Scipy inverse\n', linalg.inv(x))

## helper function
print('zeros', np.zeros((7)))
print('zeros(3x2)\n', np.zeros([3,2]))
print('\neye\n', np.eye(3))

## arrage and linspace
x = np.arange(0,2,0.1)
print(x)
x = np.linspace(0,2,20)
print(x)


import matplotlib.pyplot as plt

a = np.array([6,3,5,2,4,1])
plt.plot(a)
plt.plot([1,4,2,5,3,6])

plt.figure()
plt.plot(np.arange(0,1,0.1), [1,4,3,2,6,4,7,3,4,5])

## 0.7.1 Exercise - Create arrays
x = np.array([0.1]*10)

print(x)


## ndarray

def one_tenth(x):
    x = np.asarray(x) # convert anything to ndarray
    return x / 10.

print(one_tenth([1,2,3]))
print(one_tenth(np.array([4,5,6])))



