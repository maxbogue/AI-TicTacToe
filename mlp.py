from math import exp
from random import random

mult = lambda x, y: x * y

sigmoid = lambda x: 1 / (1 + exp(-x))

derivatives = {
    sigmoid: lambda x: sigmoid(x) * (1 - sigmoid(x)),
}

def zipWith(f, *ls):
    return map(lambda t: f(*t), zip(*ls))

def dot(xs, ys):
    # print(xs, ys)
    assert len(xs) == len(ys)
    return sum(zipWith(mult, xs, ys))

def pprint(array):
    print("[%s]" % "\n".join(map(str, array)))

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
        layout = list(map(lambda n: n + 1, layout[:-1])) + [layout[-1]];
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
        for i, w in enumerate(self.weights):
            if i < len(self.weights) - 1:
                a[i].insert(0, 1)
            raw = list(map(lambda r: dot(r, a[i]), w))
            activated = list(map(self.act, raw))
            v.append(raw)
            a.append(activated)
        if learning:
            return v, a
        else:
            return a[-1]
    
    def learn(self, t, v, a):
        """Performs the backpropagation learning algorithm.
        
        t is the truth list.
        v is the raw values for each node.
        a is the activation value for each node.
        
        """
        w = self.weights
        da = derivatives[self.act] # da for derivative of activation
        d = [[] for _ in range(len(w))] # d for deltas
        
        calc_bottom_delta = lambda i: (t[i] - a[-1][i]) * da(v[-1][i])
        d[-1] = list(map(calc_bottom_delta, range(len(t))))
        
        for l in reversed(range(len(w))[:-1]):
            delta = lambda i: dot([1] + d[l + 1], w[l][i]) * da(v[l+1][i])
            d[l] = list(map(delta, range(len(w[l]))))
        
        for l in range(len(w)):
            for j in range(len(w[l])):
                for i in range(len(w[l][j])):
                    w[l][j][i] = w[l][j][i] + self.lr * a[l][i] * d[l][j]
        # pprint(d)
    
    def train(self, xs, ts):
        """Train on a set of samples xs and truth vectors ts."""
        for x, t in zip(xs, ts):
            v, a = self.run(x, True)
            self.learn(t, v, a)
    

if __name__ == "__main__":
    m = MLP.create(2, 10, 2)
    m.train([[1, 2] for _ in range(100)], [[0,1] for _ in range(100)])
    print(m.run([1, 2]))
