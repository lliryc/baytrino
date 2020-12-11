import numpy as np
import typing as t

class ValueVector:
    def __init__(self, probs, index:int=-1):
        self.probs = np.array(probs)
        self.index = index

    def size(self):
        return len(self.probs)

    def __hash__(self):
        return hash(tuple([tuple(self.probs), self.index]))

