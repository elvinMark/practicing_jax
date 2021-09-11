import jax.numpy as jnp
from jax import grad, random

key = random.PRNGKey(0)

x_data = 10*(random.uniform(key,(100,2)) - 0.5)
y_data = 1.0*((2*x_data[:,0] + 5) > x_data[:,1])

def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))

def predict(params,x):
    W,b = params
    # return jnp.tanh(jnp.dot(x,W) + b)
    return sigmoid(jnp.dot(x,W) + b)

def loss(params):
    return 0.5*jnp.sum((predict(params,x_data) - y_data)**2)

grad_loss = grad(loss)

W = random.normal(key,(2,))
b = random.normal(key,(1,))
params = [W,b]

lr = 0.5

for epoch in range(500):
    dparams = grad_loss(params)
    for i in range(len(params)):
        params[i] -= lr * dparams[i]

pred = predict(params,x_data)

for x1,x2 in zip(y_data,pred):
    print(x1,x2)
