import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, random

key = random.PRNGKey(0)
key,W_key, b_key = random.split(key,3)

W,b = random.normal(W_key,(1,)),random.normal(b_key,(1,))

x_data = jnp.linspace(-4,4,100)
y_data = 5*x_data + 8 + 5*random.normal(key,x_data.shape)


def predict(x,W,b):
    return W*x + b

def loss(W,b):
    return 0.5*jnp.sum((predict(x_data,W,b) - y_data)**2)

lr = 0.001
grad_loss = grad(loss,(0,1))

for i in range(50):
    grad_W, grad_b = grad_loss(W,b)
    W -= lr*grad_W
    b -= lr*grad_b

print(W,b)
pred = predict(x_data,W,b)
plt.scatter(x_data, y_data)
plt.plot(x_data,pred)
plt.show()
