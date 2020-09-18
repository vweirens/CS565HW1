# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:11:01 2020

@author: Vaughn Weirens
"""

from collections import defaultdict, Counter
import itertools
import math
import random
import re


class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."

    def __init__(self):
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs

    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self


class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."

    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT

    def __repr__(self): return self.__name__


class Factor(dict): "An {outcome: frequency} mapping."


class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping.
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)


class Evidence(dict):
    "A {variable: value} mapping, describing what we know for sure."


class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."

    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table.
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple):
                row = (row,)
            self[row] = ProbDist(dist)


class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'


T = Bool(True)
F = Bool(False)


def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]


def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist


def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random() # r is a random point in the probability distribution
    c = 0.0             # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome


def globalize(mapping):
    "Given a {name: value} mapping, export all the names to the `globals()` namespace."
    globals().update(mapping)


def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})


def all_rows(net): return itertools.product(*[var.domain for var in net.variables])


def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]


def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result


def enumeration_ask(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i = net.variables.index(X) # The index of the query variable X in the row
    dist = defaultdict(float)     # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)


def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)


# %% my code here

prompt = input('Please input the prompt. (Nodes are IW, '
               + 'B, SM, R, I, G, S, M, values are true and false): ')

# extract probabilities from config.txt
saved_probs = []
with open('config.txt', 'r') as file:
    for line in file.readlines():
        float_list = [float(i) for i in line.split(',') if i.strip()]
        saved_probs += float_list

IWprob = saved_probs[0]
Bprob = saved_probs[1:3]
SMprob = saved_probs[3:5]
Rprob = saved_probs[5:7]
Iprob = saved_probs[7:9]
Gprob = saved_probs[9]
Sprob = saved_probs[10:19]
Mprob = saved_probs[19:21]

CarStartNet = (BayesNet()
               .add('IW', [], IWprob)
               .add('G', [], Gprob)
               .add('B', ['IW'], {T: Bprob[0], F: Bprob[1]})
               .add('R', ['B'], {T: Rprob[0], F: Rprob[1]})
               .add('I', ['B'], {T: Iprob[0], F: Iprob[1]})
               .add('SM', ['IW'], {T: SMprob[0], F: SMprob[1]})
               .add('S', ['I', 'SM', 'G'], {(T, T, T): Sprob[0],
                                            (T, T, F): Sprob[1],
                                            (T, F, T): Sprob[2],
                                            (F, T, T): Sprob[3],
                                            (T, F, F): Sprob[4],
                                            (F, T, F): Sprob[5],
                                            (F, F, T): Sprob[6],
                                            (F, F, F): Sprob[7]})
               .add('M', ['S'], {T: Mprob[0], F: Mprob[1]}))
CarStartJointDist = joint_distribution(CarStartNet)
globalize(CarStartNet.lookup)

query_dict = dict()

# Process query
if re.search('IW = ', prompt) is not None:
    if (re.search('IW = true', prompt) is not None or
       re.search('IW = True', prompt) is not None):
        query_dict.update({IW: True})
    elif (re.search('IW = false', prompt) is not None or
          re.search('IW = False', prompt) is not None):
        query_dict.update({IW: False})

if re.search('B = ', prompt) is not None:
    if (re.search('B = true', prompt) is not None or
       re.search('B = True', prompt) is not None):
        query_dict.update({B: True})
    elif (re.search('B = false', prompt) is not None or
          re.search('B = False', prompt) is not None):
        query_dict.update({B: False})

if re.search('SM = ', prompt) is not None:
    if (re.search('SM = true', prompt) is not None or
       re.search('SM = True', prompt) is not None):
        query_dict.update({SM: True})
    elif (re.search('SM = false', prompt) is not None or
          re.search('SM = False', prompt) is not None):
        query_dict.update({SM: False})

if re.search('R = ', prompt) is not None:
    if (re.search('R = true', prompt) is not None or
       re.search('R = True', prompt) is not None):
        query_dict.update({R: True})
    elif (re.search('R = false', prompt) is not None or
          re.search('R = False', prompt) is not None):
        query_dict.update({R: False})

if re.search('I = ', prompt) is not None:
    if (re.search('I = true', prompt) is not None or
       re.search('I = True', prompt) is not None):
        query_dict.update({I: True})
    elif (re.search('I = false', prompt) is not None or
          re.search('I = False', prompt) is not None):
        query_dict.update({I: False})

if re.search('G = ', prompt) is not None:
    if (re.search('G = true', prompt) is not None or
       re.search('G = True', prompt) is not None):
        query_dict.update({G: True})
    elif (re.search('G = false', prompt) is not None or
          re.search('G = False', prompt) is not None):
        query_dict.update({G: False})

if re.search('S = ', prompt) is not None:
    if (re.search('S = true', prompt) is not None or
       re.search('S = True', prompt) is not None):
        query_dict.update({S: True})
    elif (re.search('S = false', prompt) is not None or
          re.search('S = False', prompt)is not None):
        query_dict.update({S: False})

if re.search('M = ', prompt) is not None:
    if (re.search('M = true', prompt) is not None or
       re.search('M = True', prompt) is not None):
        query_dict.update({M: True})
    elif (re.search('M = false', prompt) is not None or
          re.search('M = False', prompt) is not None):
        query_dict.update({M: False})

total_prob = 0
evidence = {IW: True, B: False, R: False}
for (row, p) in CarStartJointDist.items():
    if matches_evidence(row, query_dict, CarStartNet):
            total_prob += p
            
print('The total probability for the input query is:', total_prob)
