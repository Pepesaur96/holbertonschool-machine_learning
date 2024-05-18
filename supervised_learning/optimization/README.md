# What is a Hyperparameter?

A hyperparameter is a parameter whose value is set before the learning process begins. Unlike model parameters (weights and biases), which are learned during training, hyperparameters are predefined and control the learning process. Examples include learning rate, batch size, number of epochs, and the architecture of the model (like the number of layers and units in each layer).

# How and Why Do You Normalize Your Input Data?

Normalization is the process of scaling individual samples to have a mean of zero and a standard deviation of one. This ensures that each feature contributes equally to the learning process and prevents features with larger values from dominating.

- Why Normalize?

  - It helps in faster convergence during training.
  - It improves the stability and performance of the model.
  - It reduces the chances of getting stuck in local minima.

- How to Normalize?

  - Subtract the mean of the data from each feature.
  - Divide each feature by its standard deviation.

  Example in Python using NumPy:

      import numpy as np

      def normalize(data):
          mean = np.mean(data, axis=0)
          std = np.std(data, axis=0)
          normalized_data = (data - mean) / std
          return normalized_data

# What is a Saddle Point?

A saddle point is a point in the loss landscape of a neural network where the gradient is zero but is not a local minimum or maximum. It can be a flat region or a point where the function curves up in one direction and down in another. Saddle points can slow down the training process because gradients near these points are small, leading to slow updates.

# What is Stochastic Gradient Descent (SGD)?

SGD is an iterative method for optimizing an objective function. In standard gradient descent, the entire dataset is used to compute the gradient of the loss function, which can be computationally expensive. SGD, on the other hand, updates the parameters using only one training example at a time.

- Advantages of SGD:

  - Faster convergence on large datasets.
  - Introduces noise into the gradient calculation, which can help escape local minima.

  Example in Python:

      def sgd(X, y, params, learning_rate):
      for i in range(len(y)):
      gradients = compute_gradients(X[i], y[i], params)
      for param in params:
      params[param] -= learning_rate \* gradients[param]

# What is Mini-Batch Gradient Descent?

Mini-batch gradient descent is a compromise between full batch gradient descent and stochastic gradient descent. Instead of using the entire dataset or just one example, mini-batch gradient descent uses a small random subset (mini-batch) of the data to compute the gradient and update the parameters.

- Advantages:

  - Reduces variance of the parameter updates, leading to more stable convergence.
  - Can be more efficient than SGD due to optimized matrix operations.

  Example in Python:

      def mini_batch_gd(X, y, params, learning_rate, batch_size):
      for i in range(0, len(y), batch_size):
      X_batch = X[i:i + batch_size]
      y_batch = y[i:i + batch_size]
      gradients = compute_gradients(X_batch, y_batch, params)
      for param in params:
      params[param] -= learning_rate \* gradients[param]

# What is a Moving Average? How Do You Implement It?

A moving average smoothens a series by averaging values within a moving window. It's often used in time series analysis and to smooth training metrics in machine learning.

Simple Moving Average (SMA) Implementation:

    def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# What is Gradient Descent with Momentum? How Do You Implement It?

Momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging.

Implementation:

    def gradient*descent_with_momentum(X, y, params, learning_rate, momentum, velocity):
    gradients = compute_gradients(X, y, params)
    for param in params:
    velocity[param] = momentum * velocity[param] - learning*rate * gradients[param]
    params[param] += velocity[param]

# What is RMSProp? How Do You Implement It?

RMSProp (Root Mean Square Propagation) adapts the learning rate for each of the parameters. It keeps a moving average of the squared gradients for each parameter and divides the gradient by the root of this average.

Implementation:

    def rmsprop(X, y, params, learning*rate, beta, epsilon, cache):
    gradients = compute_gradients(X, y, params)
    for param in params:
    cache[param] = beta * cache[param] + (1 - beta) _ gradients[param]\*\_2
    params[param] -= learning_rate _ gradients[param] / (np.sqrt(cache[param]) + epsilon)

# What is Adam Optimization? How Do You Implement It?

Adam (Adaptive Moment Estimation) combines the advantages of both RMSProp and Momentum. It maintains a moving average of both the gradients and the squared gradients.

Implementation:

    def adam(X, y, params, learning*rate, beta1, beta2, epsilon, t, m, v):
    gradients = compute_gradients(X, y, params)
    t += 1
    for param in params:
    m[param] = beta1 * m[param] + (1 - beta1) _ gradients[param]
    v[param] = beta2 _ v[param] + (1 - beta2) _ gradients[param]**2
    m_hat = m[param] / (1 - beta1**t)
    v_hat = v[param] / (1 - beta2\*\_t)
    params[param] -= learning_rate _ m_hat / (np.sqrt(v_hat) + epsilon)

# What is Learning Rate Decay? How Do You Implement It?

Learning rate decay reduces the learning rate over time to ensure the training converges smoothly.

Implementation:

    def learning_rate_decay(initial_lr, decay_rate, epoch):
    return initial_lr / (1 + decay_rate \* epoch)

# What is Batch Normalization? How Do You Implement It?

Batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation, then scaling and shifting.

Implementation:

    def batch_norm(X, gamma, beta, epsilon):
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    out = gamma \* X_normalized + beta
    return out

Each of these methods and techniques contributes to the efficiency and effectiveness of training deep neural networks, ensuring faster convergence, better generalization, and more stable training dynamics.
