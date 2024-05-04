1. What is a Model?
   In machine learning, a model is a mathematical representation of a real-world process. A machine learning model is trained from data and is used to make predictions or decisions without being explicitly programmed to perform the task.
2. What is Supervised Learning?
   Supervised learning is a type of machine learning where the model is trained on labeled data. The model learns to map an input to an output based on input-output pairs, using those to predict the output from new inputs.
3. What is a Prediction?
   A prediction is the output produced by a model when given an input. In the context of machine learning, a prediction is typically the result of processing new data through a trained model.
4. What is a Node?
   In various contexts, a node can mean different things. In neural networks, a node is a neuron-like unit that receives inputs, processes them, and passes the output to the next layer. In decision trees, a node represents a point where the data is split according to a specific feature.
5. What is a Weight?
   Weights are the parameters within a neural network that transform input data within the network's architecture. They are adjusted during training to minimize the network's loss function.
6. What is a Bias?
   Bias is another parameter in a neural network that allows the model to represent patterns that do not pass through the origin. By adjusting the bias, the model can better fit the data.
7. What are Activation Functions?
   Activation functions determine the output of a neural network node given a set of inputs. They introduce non-linear properties to the network, allowing it to learn more complex patterns.
8. Common Activation Functions:
   Sigmoid: A function that maps any real-valued number into the range (0, 1), making it useful for models where we need to predict the probability as an output.
   Tanh (Hyperbolic Tangent): Similar to the sigmoid but maps real-valued numbers into the range (-1, 1).
   ReLU (Rectified Linear Unit): Allows only positive values to pass through it, which makes it efficient and effective for many non-linear problems.
   Softmax: Used in multi-class classification, this function converts a vector of values into a probability distribution.
9. What is a Layer?
   A layer is a collection of nodes (neurons) operating together at a specific depth within a neural network.
10. What is a Hidden Layer?
    Hidden layers are layers between the input layer and the output layer in a neural network. These layers perform computations and feature extractions.
11. What is Logistic Regression?
    Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist.
12. What is a Loss Function?
    A loss function quantifies the difference between the expected outcomes and the outcomes predicted by the model. It guides the training algorithms to optimize the model.
13. What is a Cost Function?
    Often used interchangeably with a loss function, in the context of machine learning, it usually refers to the average loss over the entire training dataset.
14. What is Forward Propagation?
    Forward propagation is the process of moving input data through the neural network to generate an output.
15. What is Gradient Descent?
    Gradient descent is an optimization algorithm used to minimize the function (such as a loss function) by iteratively moving towards the steepest descent as defined by the negative of the gradient.
16. What is Back Propagation?
    Back propagation is the method of refining the model’s weights by propagating the error back through the network, calculating the gradient and then applying gradient descent to update the weights.
17. What is a Computation Graph?
    A computation graph is a graphical representation of the operations and dependencies between individual steps in a model, which is useful for visualizing the flow of computations and for performing automatic differentiation.
18. How to Initialize Weights/Biases
    Weights and biases should be initialized not too small to avoid slow convergence, and not too large to avoid overshooting the minimum during training. Common practices include initializing weights with small random numbers and biases to zero or small constants.
19. The Importance of Vectorization
    Vectorization involves expressing computational operations as matrix and vector operations. It speeds up the data processing by eliminating explicit loops in code, leveraging optimizations in linear algebra libraries.
20. How to Split Up Your Data
    Data is typically split into training, validation, and test sets. The training set is used to train the model, the validation set to adjust parameters and prevent overfitting, and the test set to evaluate the model’s performance.
21. What is Multiclass Classification?
    Multiclass classification refers to classifying instances into one of three or more classes, as opposed to binary classification, which involves two classes.
22. What is a One-Hot Vector?
    A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension, used to categorically represent the presence of a specific value from a set of discrete classes.
23. How to Encode/Decode One-Hot Vectors
    Encoding involves converting categorical labels into vectors from 0s and a single 1. Decoding is the reverse, converting these vectors back to categorical labels.
24. What is the Softmax Function and When Do You Use It?
    The softmax function is used in multinomial logistic regression and is effective in multiclass classification tasks. It converts a vector of values into a probability distribution.
25. What is Cross-Entropy Loss?
    Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. It increases as the predicted probability diverges from the actual label.
26. What is Pickling in Python?
    Pickling is a way to serialize and deserialize a Python object structure. Any object in Python can be pickled so that it can be saved on disk.
