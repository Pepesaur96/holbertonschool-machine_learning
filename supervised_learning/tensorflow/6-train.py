#!/usr/bin/env python3
""" Module that builds, trains, and saves a neural network classifier"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    Args:
        X_train: numpy.ndarray - Contains the training input data.
        Y_train: numpy.ndarray - Contains the training labels.
        X_valid: numpy.ndarray - Contains the validation input data.
        Y_valid: numpy.ndarray - Contains the validation labels.
        layer_sizes: list - Contains the number of nodes in each
                            layer of the network.
        activations: list - Contains the activation functions for each
                            layer of the network.
        alpha: float - Learning rate.
        iterations: int - Number of iterations to train over.
        save_path: str - Designates where to save the model.
    Returns:
        str - The path where the model was saved.
    """
    # Import necessary functions
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_placeholders = __import__(
        '0-create_placeholders').create_placeholders
    create_train_op = __import__('5-create_train_op').create_train_op
    forward_prop = __import__('2-forward_prop').forward_prop

    # Set random seed for reproducibility
    tf.set_random_seed(0)

    # Reset the graph
    tf.reset_default_graph()

    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the forward propagation model
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Add important tensors to the graph's collection
    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)
    tf.add_to_collection('outputs', y_pred)
    tf.add_to_collection('losses', loss)
    tf.add_to_collection('accuracies', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Create saver object to save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Run the training operation and calculate loss and accuracy on the training data
            _, train_loss, train_acc = sess.run(
                [train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                # Calculate loss and accuracy on the validation data
                valid_loss, valid_acc = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_acc}")

        # Save the model to the specified path
        save_path = saver.save(sess, save_path)
        return save_path
