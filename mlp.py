from random import random

class MLP(object):
    
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
                for k in range(layout[i+1])]
                    for j in range(layout[i])]
                        for i in range(len(layout) - 1)]
        return MLP(weights, **extras)
    
    def __init__(self, weights, lr=0.1):
        """Initializes an MLP.
        
        weights is a list of weight arrays for each layer.
        lr is the learning rate.
        
        """
        self.weights = weights
        self.lr = lr
    
    def run(self, x):
        """Runs the MLP on input list x."""
        pass

print MLP.create(3,2,1).weights
