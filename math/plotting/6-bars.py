#!/usr/bin/env python3
"""
Function that creates a bar graph with multiple bars, one for each member of a group
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Function that creates a bar graph with multiple bars, one for each member of a group
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']

    fruits = [('apples', 'red'), ('bananas', 'yellow'),
              ('oranges', 'orange'), ('peaches', 'pink')]

    bottom = np.zeros(len(names))

    for fruit_name, color in fruits:
        plt.bar(names, fruit[fruits.index((fruit_name, color))],
                bottom=bottom, color=color, width=0.5)

        bottom += fruit[fruits.index((fruit_name, color))]

    plt.legend([fruit_name for fruit_name, color in fruits])

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))

    plt.title('Number of Fruit per Person')

    plt.show()
