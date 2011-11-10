from math import exp
from random import random

def sigmoid(x):
    return 1 / (1 + exp(-x))

derivatives {
    sigmoid: lambda x: sigmoid(x) * (1 - sigmoid(x)),
}
def zipWith(f, *ls):
    return map(lambda t: f(*t), zip(*ls))

class MLP(object):
    """Represents a multi-layer perceptron.
    
    The weights are oriented as follows:
    The first dimension is the layer (w[i] is between layers i and i+1).
    The second dimension is the destination node (in layer i+1).
    The third dimension is the source node (from layer i).
    
    """
    @staticmethod
    def create(*layout, **extras):
        """Creates an MLP with random weights.
        
        layout is a list of numbers of nodes for each layer; e.g.
        MLP.create(10, 10, 1) produces 10 input nodes, 10 hidden nodes,
        and 1 output node.
        
        """
        # Add a bias for each layer except the last.
        layout = map(lambda n: n + 1, layout[:-1]) + [layout[-1]];
        # Generate weights using awesome list comprehensions.
        weights = [[[
            random() * 2 - 1
                for k in range(layout[i])]
                    for j in range(layout[i+1])]
                        for i in range(len(layout) - 1)]
        return MLP(weights, **extras)
    
    def __init__(self, weights, lr=0.1, act=sigmoid):
        """Initializes an MLP.
        
        weights is a list of weight arrays for each layer.
        lr is the learning rate.
        
        """
        self.weights = weights
        self.lr = lr
        self.act = act
    
    def run(self, x, learning=False):
        """Runs the MLP on input list x.
        
        Returns the output if learning=False and a tuple of raw and
        activated values matrices if it's True.
        
        """
        v = [x] # raw values
        a = [x] # activated values
        mult = lambda x, y: x * y
        for i, w in enumerate(self.weights):
            raw = map(lambda r: sum(zipWith(mult, r, a[i])), self.weights)
            v.append(list(raw))
            a.append(list(map(self.act, raw)))
        if learning:
            return v, a
        else:
            return a[-1]
    

