import numpy as np
import typing as t

class ProbRelation:
    def __init__(self, probs: t.List[t.List[float]], shape:t.Tuple[int, int]):
        self.distibution = np.array(probs)
        self.shape = shape