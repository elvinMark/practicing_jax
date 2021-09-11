import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax import random
import numpy as np

def f(x):
    return np.sum(x**2)


grad_f = grad(f)

x = 3.
print(f"x = {x}")
print(f"f(x)= {f(x)}")
print(f"df(x)= {grad_f(x)}")
