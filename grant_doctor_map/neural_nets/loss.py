"""Loss function is the same as an error function and is nothing more
than the difference between where I am now and where I want to be.
"""

import numpy as np
from grant_doctor_map import tensor


class Loss():
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        """The value of the loss function is the difference of
        predictions and the known labels.

        Args:
            predictions (tensor.Tensor): predicted values from model
            labels (tensor.Tensor): known data

        Returns:
            float: the value of the loss
        """
        raise NotImplementedError
    
    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        """The gradient of the loss function with respect to the 
        predictions
        
        Args:
            predictions (tensor.Tensor): predicted values from model
            labels (tensor.Tensor): known data

        Returns:
            tensor.Tensor: the same size as the predictions

        """
        raise NotImplementedError
    

class MSE(Loss):
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        return np.mean((predictions - labels)**2)
    
    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        return 2*(predictions - labels)