import numpy as np
import typing as t
from  discrete_distribution import DiscreteDistribution
from conditional_distribution import ProbRelation
#import dask
#import dask.array as da
#dask.config.set(scheduler='threads')

class BayesianNet:

    def __init__(self):

        self.nodes = {}
        self.edges = {}
        self.edges_inv = {}
        self.cache = {}

    def add_node(self, name:str, distr:t.List[float]):
        if str is self.nodes:
            raise Exception(f"Node with name '{name}' already exists")
        self.nodes[name] = DiscreteDistribution(distr)
        self.edges[name] = {}

    def add_edge(self, ancestor: str, descendent: str, conditional_prob_matrix: t.List[t.List[float]]): # P(a|d)
        if ancestor not in self.nodes:
            raise Exception(f"Node with name '{ancestor}' doesn't exist yet")
        if descendent not in self.nodes:
            raise Exception(f"Node with name '{descendent}' doesn't exist yet")
        if descendent in self.edges[ancestor]:
            raise Exception(f"The edge ('{ancestor}','{descendent}') already exists")
        cpd = ProbRelation(conditional_prob_matrix, (len(conditional_prob_matrix), len(conditional_prob_matrix[0])))
        self.edges[ancestor][descendent] = cpd


    def get_evidence(self, vars:t.Dict[str, int], target_var:str)->np.ndarray:
        self.edges_inv = {}

        for ancestor in self.edges:
            for descendent in self.edges[ancestor]:
                if descendent not in self.edges_inv:
                    self.edges_inv[descendent] = {}
                self.edges_inv[descendent][ancestor] = self.edges[ancestor][descendent]

        self.cache = dict(map(lambda var_key: (var_key, DiscreteDistribution([float(i == vars[var_key]) for i in range(0, self.nodes[var_key].size())], vars[var_key])),vars.keys()))
        return self._compute_var(target_var).probs.tolist()

    def _calc_di(self, cpd:ProbRelation, ancestor_key, ancestor_distr:DiscreteDistribution, child_key)->ProbRelation:
        if ancestor_distr.index != -1:
            cpd_mx: np.ndarray = cpd.distibution[ancestor_distr.index].reshape(1, cpd.shape[1])
            ancestor_vec: np.ndarray = np.array([self.nodes[ancestor_key].probs[ancestor_distr.index]])
            ancestor_size = len(ancestor_vec)
        else:
            cpd_mx: np.ndarray = cpd.distibution
            ancestor_vec: np.ndarray = self.nodes[ancestor_key].probs
            ancestor_size = len(ancestor_vec)
        child_vec: np.ndarray = self.nodes[child_key].probs
        child_size = len(child_vec)
        denom = cpd_mx * child_vec.reshape(1,child_size)
        result = (ancestor_vec.reshape(ancestor_size, 1) / denom) - np.array([1.0])
        return ProbRelation(result, result.shape)

    def _mult_pair_di(self, d1:ProbRelation, d2:ProbRelation)->ProbRelation:
        d1_splits = np.hsplit(d1.distibution, d1.shape[1])
        d2_splits = np.hsplit(d2.distibution, d2.shape[1])

        d_splits = [np.ndarray.reshape(d1_splits[k].reshape(d1.shape[0],1) * d2_splits[k].reshape(1, d2.shape[0]), (d1.shape[0] * d2.shape[0], 1)) for k in range(0, d1.shape[1])]
        assembly = np.hstack(d_splits)
        return ProbRelation(assembly, assembly.shape)

    def _calc_a(self, child_key)->np.ndarray:
        vec = self.nodes[child_key].probs
        complement = np.array([1.0]) - vec
        return complement / vec

    def _compute_var(self, var:str)->DiscreteDistribution:
        if var in self.cache:
            return self.cache[var]
        if var not in self.edges_inv:
            return None

        cache_records = dict(filter(lambda tuple: tuple[1] is not None, map(lambda ancestor_key: (ancestor_key, self._compute_var(ancestor_key)), self.edges_inv[var])))
        if len(cache_records) == 0:
            return None
        dis = list(map(lambda ancestor_key: self._calc_di(self.edges[ancestor_key][var], ancestor_key, cache_records[ancestor_key],var),cache_records))
        len_dis = len(dis)

        while(len_dis > 1):
            if len_dis % 2 == 1:
                dis.append(None)
            ixs = range(0, len(dis)//2)
            dis = list(map(lambda i: self._mult_pair_di(dis[i], dis[i+1]), ixs))
            len_dis = len(dis)

        a = np.power(self._calc_a(var), len(cache_records) - 1)
        x = dis[0].distibution / a
        result = np.array([1]) / (np.array([1]) + x)
        result = np.sum(result, axis=0)
        mx = np.max(result)
        if mx == 1:
            distribution = DiscreteDistribution(result, np.argmax(result))
        else:
            distribution = DiscreteDistribution(result)
        self.cache[var] = distribution
        return distribution

if __name__ == '__main__':
    bnet = BayesianNet()
    bnet.add_node('Y', [0.4, 0.3, 0.3])
    bnet.add_node('X', [0.3, 0.4, 0.3])
    bnet.add_node('A', [0.5, 0.5])
    bnet.add_edge('X', 'A', [[0.2, 0.4], [0.55, 0.25], [0.25, 0.35]])
    bnet.add_edge('Y', 'A', [[0.1, 0.7], [0.4, 0.2], [0.5, 0.1]])
    pA = bnet.get_evidence({'X': 0, 'Y': 2}, 'A')
    print(pA)













