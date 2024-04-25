#!/usr/bin/env python3
""" Decision Tree """
import numpy as np


class Node:
    """
    Class that represents a decision tree node
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Class constructor for Node class
        Args:
            feature (_type_, optional): _description_. Defaults to None.
            threshold (_type_, optional): _description_. Defaults to None.
            left_child (_type_, optional): _description_. Defaults to None.
            right_child (_type_, optional): _description_. Defaults to None.
            is_root (bool, optional): _description_. Defaults to False.
            depth (int, optional): _description_. Defaults to 0.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Method that calculates the maximum depth of the current node
        Returns:
            int: maximum depth of the current node
        """
        # If the node is a leaf, its max depth is its own depth
        if not self.left_child and not self.right_child:
            return self.depth

        # Initialize depths assuming the current node is the deepest
        left_depth = self.depth
        right_depth = self.depth

        # Recursively find the max depth of the left subtree
        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()

        # Recursively find the max depth of the right subtree
        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        # Return the maximum of left and right depths
        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Class that represents a leaf node in a decision tree
    """

    def __init__(self, value, depth=None):
        """
        Class constructor for Leaf class
        Args:
            value (_type_): _description_.
            depth (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Method that calculates the maximum depth of the current node
        Returns:
            int: maximum depth of the current node
        """
        return self.depth


class Decision_Tree():
    """
    Class that represents a decision tree
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Class constructor for Decision_Tree class
        Args:
            max_depth (int, optional): _description_. Defaults to 10.
            min_pop (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 0.
            split_criterion (str, optional): _description_.
                Defaults to "random".
            root (_type_, optional): _description_. Defaults to None.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Method that calculates the depth of the decision tree
        """
        return self.root.max_depth_below()
