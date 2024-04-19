"""A layer is a single set of neurons and it has to be able to 
learn via backpropagation as well as run the network forward/
feedforward
"""
from typing import Callable
import numpy as np

from grant_doctor_map import tensor


class Layer():
    def __init__(self):
        self.w = tensor.Tensor
        self.b = tensor.Tensor
        self.x = None  # Can be thought of as inputs
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute the forward pass of the neurons in a layer"""
        raise NotImplementedError
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Compute the backpropagation through the layer"""
        raise NotImplementedError
    

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        """Create a linear layer

        Args:
            input_size (int): the number of input values 
            (batch_size, input_size)
            output_size (int): the number of output values to the next
            layer or final size (batch_size, output_size)
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute y = w @ x + b
        Where @ is the matrix multiplication version of scalar *"""
        self.x = x
        return self.x @ self.w + self.b
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """We are going to have to compute the partial derivates to 
        figure out what the heck is going on.
        X = w*x + b
        y = f(x)
        dy/dw = f'(X)*x
        dy/dx = f'(X)*w
        dy/db = f'(X)

        Now let's put this in tensor form (i.e. matrix math)
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T
    

class Activation(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 f,
                 f_prime):
        """Initialize an activation layer as generic layer that has a
        forward function and its derivative for backpropagation.

        Args:
            input_size (int): the number of input values 
                (batch_size, input_size)
            output_size (int): the number of output values to the next
                layer or final size (batch_size, output_size)
            f: a function
            f_prime: the derivative of f
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.f = f
        self.f_prime = f_prime

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x
        return self.f(self.x @ self.w + self.b)
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        grad = grad @ self.w.T
        return self.f_prime(self.x)*grad
    

def tanh(x: tensor.Tensor) -> tensor.Tensor:
    return np.tanh(x)

def tanh_prime(x: tensor.Tensor) -> tensor.Tensor:
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, tanh, tanh_prime)