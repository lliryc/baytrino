import numpy as np
import typing as t

class DiscreteDistribution:
    def __init__(self, probs, index:int=-1):
        self.probs = np.array(probs)
        self.index = index
    def size(self):
        return len(self.probs)
