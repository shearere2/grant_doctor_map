"""Create a mechanism for batching data for our neural network"""
import numpy as np
from typing import Iterator

from grant_doctor_map import tensor


class DataIterator():
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        """Batch a set of features and labels
        
        Args:
            features (tensor.Tensor): features
            labels (tensor.Tensor): associated labels per row
            
        Yields:
            Iterator: a subset of data to be trained upon (batch)"""
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self,batch_size: int = 32, shuffle: bool = True):
        """Create a new iterator for batching data

        Args:
            batch_size (int, optional): the number of values in batch.
            Defaults to 32.
            shuffle (bool, optional): Whether to shuffle training data.
            Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        starts = np.arange(0,len(features),self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_features = features[start:end]
            batch_labels = labels[start:end]
            yield (batch_features,batch_labels)