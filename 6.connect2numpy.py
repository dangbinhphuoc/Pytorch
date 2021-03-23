import torch
import numpy as np

# torch -> numpy
x = torch.ones(5)
print(type(x))

y = x.numpy()
print(type(y))

print()
print(x)
print(y)

print()
x.add_(1)
print(x)
print(y)

print()

# numpy -> torch
x = np.ones(5)
y = torch.from_numpy(x)
print(type(x))
print(type(y))
print(x)
print(y)