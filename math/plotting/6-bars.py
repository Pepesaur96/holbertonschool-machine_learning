#!/usr/bin/env python3
"""
This module generates a stacked bar graph
representing the quantity of different fruits
possessed by Farrah, Fred, and Felicia.
The fruits are represented by different colors.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    This function generates a stacked bar graph
    representing the quantity of different fruits
    possessed by Farrah, Fred, and Felicia.
    The fruits are represented by different colors.
    """
    # Set the seed for the random number generator for reproducibility
    np.random.seed(5)

    # Generate a 4x3 matrix of random integers between 0 and 20
    # Each column represents a person (Farrah, Fred, Felicia)
    # Each row represents a type of fruit (apple, banana, orange, peach)
    fruit = np.random.randint(0, 20, (4, 3))

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Set the names of the people
    names = ['Farrah', 'Fred', 'Felicia']

    # Set the names of the fruits and their corresponding colors
    fruits = [('apples', 'red'), ('bananas', 'yellow'),
              ('oranges', '#ff8000'), ('peaches', '#ffe5b4')]

    # Initialize the bottom of the bars (needed for stacking the bars)
    bottom = np.zeros(len(names))

    # Loop over each type of fruit
    for fruit_name, color in fruits:
        # Create a bar for the current type of fruit
        # The height of the bar is the number of this type of fruit
        # The bottom of the bar is the top of the previous bar
        # (or 0 for the first bar)
        plt.bar(names, fruit[fruits.index((fruit_name, color))],
                bottom=bottom, color=color, width=0.5)

        # Update the bottom for the next bar
        bottom += fruit[fruits.index((fruit_name, color))]

    # Add a legend to the plot
    plt.legend([fruit_name for fruit_name, color in fruits])

    # Label the y-axis and set its range and tick marks
    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))

    # Set the title of the plot
    plt.title('Number of Fruit per Person')

    # Display the plot
    plt.show()
