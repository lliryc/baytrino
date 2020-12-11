import numpy as np
import typing as t
from  value_vector import ValueVector
from conditional_distribution import ProbRelation
import functools
import random
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, Inference
from pgmpy.sampling import BayesianModelSampling
import time

import pgmpy
#import dask
#import dask.array as da
#dask.config.set(scheduler='threads')

class BayesianNet:

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.edges_inv = {}
        self.var_cache = {}
        self.factor_cache = {}

    def add_node(self, name:str, distr:t.List[float]):
        if str is self.nodes:
            raise Exception(f"Node with name '{name}' already exists")
        self.nodes[name] = ValueVector(distr)
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

    def get_evidence(self, vars:t.Dict[str, int], target_var:str):
        self.edges_inv = {}

        for ancestor in self.edges:
            for descendent in self.edges[ancestor]:
                if descendent not in self.edges_inv:
                    self.edges_inv[descendent] = {}
                self.edges_inv[descendent][ancestor] = self.edges[ancestor][descendent]

        self.var_cache = dict(map(lambda var_key: (var_key, ValueVector([ float(i==vars[var_key]) for i in range(0, self.nodes[var_key].size())], index=vars[var_key])), vars.keys()))
        probs = self._compute_var(target_var).probs
        denom = np.sum(probs)
        return (probs / denom).tolist()

    def _calc_di(self, cpd:ProbRelation, ancestor_key, ancestor_distr:ValueVector, child_key)->ProbRelation:

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

    def _compute_var(self, var:str)->ValueVector:
        if var in self.var_cache:
            return self.var_cache[var]

        if var not in self.edges_inv:
            return None

        ancestor_vars = self.edges_inv[var]
        kv_anc = map(lambda ancestor_key: (ancestor_key, self._compute_var(ancestor_key)), ancestor_vars)
        kv_anc = list(filter(lambda tuple: tuple[1] is not None, kv_anc))
        var_vec = self._compute_factor(var, *kv_anc)
        probs = (var_vec.probs * self.nodes[var].probs)
        self.var_cache[var] = ValueVector(probs)
        return ValueVector(probs)

    def __hash__(self):
        return hash(repr(self))

    @staticmethod
    def _apply_cpd(v:ValueVector, cpd: ProbRelation):
        if v.index != -1:
            rel = cpd.distibution[v.index].reshape(1, cpd.distibution.shape[1])
            return ProbRelation(rel, rel.shape)
        else:
            rel = v.probs.reshape(v.size(), 1) * cpd.distibution
            return ProbRelation(rel, rel.shape)

    @staticmethod
    def _mult_pair_cpd(cpd1: ProbRelation, cpd2: ProbRelation)->ProbRelation:
        if(cpd2 is None):
            return cpd1
        d1_splits = np.hsplit(cpd1.distibution, cpd1.shape[1])
        d2_splits = np.hsplit(cpd2.distibution, cpd2.shape[1])

        d_splits = [np.ndarray.reshape(d1_splits[k].reshape(cpd1.shape[0],1) * d2_splits[k].reshape(1, cpd2.shape[0]), (cpd1.shape[0] * cpd2.shape[0], 1)) for k in range(0, cpd1.shape[1])]
        assembly = np.hstack(d_splits)
        return ProbRelation(assembly, assembly.shape)


    def _compute_factor(self, var_child, *var_ancestors)->ValueVector:
        h = hash(tuple([var_child, var_ancestors]))

        if h in self.factor_cache:
            return self.factor_cache[h]

        anc_dict = dict(var_ancestors)
        anc_rels = list(map(lambda var_anc: self._apply_cpd(anc_dict[var_anc], self.edges[var_anc][var_child]), anc_dict))
        len_rels = len(anc_rels)

        while(len_rels > 1):
            if len_rels % 2 == 1:
                anc_rels.append(None)
                len_rels = len_rels + 1
            anc_rels = list(map(lambda i: self._mult_pair_cpd(anc_rels[i], anc_rels[i+1]), range(0, len_rels//2)))
            len_rels = len(anc_rels)
        dist = anc_rels[0].distibution
        vec = dist.sum(axis=0)
        res = ValueVector(vec)
        self.factor_cache[h] = res
        return res

def run_baytrino_model():
    bnet = BayesianNet()
    bnet.add_node('Y', [0.4, 0.3, 0.3])
    bnet.add_node('X', [0.3, 0.4, 0.3])
    bnet.add_node('A', [0.5, 0.5])
    bnet.add_edge('X', 'A', [[0.2, 0.4], [0.55, 0.25], [0.25, 0.35]])
    bnet.add_edge('Y', 'A', [[0.1, 0.7], [0.4, 0.2], [0.5, 0.1]])
    pA = bnet.get_evidence({'X': 0, 'Y': 2}, 'A')
    print(pA)

def test1_baytrino():
    start_time = time.time()

    bnet = BayesianNet()
    bnet.add_node('A', [0.5, 0.5])
    vals = {}
    vals_ix = {}
    for i in range(0, 100):
        var = 'X' + str(i)
        bnet.add_node(var, [0.3, 0.4, 0.3])
        bnet.add_edge(var, 'A', [[0.2, 0.4], [0.55, 0.25], [0.25, 0.35]])
        vals[var] = 0
        vals_ix[i] = var

    for i in range(0, 1000):
        index = random.randrange(0,100)
        if(vals[vals_ix[index]] == 1):
            vals[vals_ix[index]] = 0
        else:
            vals[vals_ix[index]] = 1
        bnet.get_evidence(vals, 'A')

    end_time = time.time()
    return end_time - start_time

def run_pgmpy_model():
    edges = []
    cdp_vars = []
    cdp_a = TabularCPD('A', 2, [[0.5], [0.5]])
    cdp_vars.append(cdp_a)

    cdp_x = TabularCPD('X', 3, [[0.2, 0.4], [0.55, 0.25], [0.25, 0.35]], ['A'], [2])
    cdp_vars.append(cdp_x)
    edges.append(('A', 'X'))

    cdp_y = TabularCPD('Y', 3, [[0.1, 0.7], [0.4, 0.2], [0.5, 0.1]], ['A'], [2])
    cdp_vars.append(cdp_y)
    edges.append(('A', 'Y'))

    bmodel = BayesianModel(edges)
    bmodel.add_cpds(*cdp_vars)

    inf = VariableElimination(bmodel)

    inf.query(['A'], {'X': 0, 'Y': 2})

def test1_pgmpy():
    start_time = time.time()

    edges = []
    cdp_vars = []
    cdp_a = TabularCPD('A', 2, [[0.5], [0.5]])
    cdp_vars.append(cdp_a)
    vals = {}

    for i in range(0,100):
        var  = 'X' + str(i)
        cdp_x = TabularCPD(var, 3, [[0.2, 0.4], [0.55, 0.25], [0.25, 0.35]], ['A'], [2])
        cdp_vars.append(cdp_x)
        edges.append(('A', var))
        vals[var] = 0

    bmodel = BayesianModel(edges)
    bmodel.add_cpds(*cdp_vars)
    inf = VariableElimination(bmodel)
    for i in range(0, 1000):
        (inf.query(['A'], vals))
    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    #print(f"test1 brazil, time elapsed '{test1_pgmpy()}'")
    print(f"test1 baytrino, time elapsed '{test1_baytrino()}'")

















