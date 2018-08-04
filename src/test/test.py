import numpy as np


print('Hello World')


x = np.array([[0, 1, 2, 3], [5, 3, 2, 1]])
y = np.ones((2, 4), dtype=int)
z = np.arange(4)[np.newaxis]
y[0] = 0
print(f'x: \n{x}')
print(f'y: \n{y}')
print(f'z: \n{z}')

print(f'x*y: \n{x*y}')

z2 = z.T
print(f'z: \n{z2}')


#print(f'x*z: \n{x.dot(z)}')
print(f'x*z2: \n{x.dot(z2)}')
