"""An MLP or neural network or fully connected neural network or
deep neural network is a collection of layers that pass information
forward.
"""
from grant_doctor_map import tensor, layer


class MLP():
    def __init__(self, layers: list[layer.Layer]):
        self.layers = layers

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute a forward pass through the entire network"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Compute backprop pass through the entire network"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self):
        """Return the weights and biases for every single layer in turn
        along with their gradients"""
        for layer in self.layers:
            for pair in [(layer.w, layer.grad_w), (layer.b, layer.grad_b)]:
                yield pair

    def zero_parameters(self):
        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0