#!/usr/bin/env python3
"""
This module provides a Random Forest classifier that uses Decision Trees
as its base learners. It includes methods for fitting the model to training
data, making predictions, and calculating accuracy.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """ This class defines a Random Forest classifier. """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Initializes the Random Forest classifier. """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """ Makes predictions for a given set of examples. """
        predictions = []

        # Generate predictions for each tree in the forest
        for predict_function in self.numpy_preds:
            predictions.append(predict_function(explanatory))

        predictions = np.array(predictions)

        # Calculate the mode (most frequent) prediction for each example
        mode_predictions = []
        for example_predictions in predictions.T:
            unique_values, counts = np.unique(example_predictions,
                                              return_counts=True)
            mode_index = np.argmax(counts)
            mode_predictions.append(unique_values[mode_index])

        return np.array(mode_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """ Fits the model to the training data. """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """ Calculates the accuracy of the model on test data. """
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target))/test_target.size
