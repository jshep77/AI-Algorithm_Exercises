# assignment5

In both parts of this assignment, implementation was very straightforward.  Each problem had it's own set of functions to complete the goal, located in utils.py.  Thankfully, each function was a small helper function to the more complex portions of the assignments, building the models.  Each function in the util.py folder includes notes on what the function returns.

The first part of the assignment was to build our own version of the K nearest neighbor classifier.  The functions used to complete this goal are:

fit(self, X, y):
 This method trains the model. It takes training data X and corresponding target values y as input and stores them in the class.

predict(self, X):
 This method is used to predict new data points. The distances between the testing sample and each training sample are calculated using either Euclidean or Manhattan distance, based on the specified weight value.  The k-nearest neighbors are determined by sorting the distances and selecting the first k indices.  Then comes the primary portion of the function, if the weights input is "uniform", the predicted label is the most commonly occuring class among the k neighbors.  If the weights input is "distance", the neighbors' input is weighted based on the inverse of their distances, and the predicted label is the one with the highest weighted sum.

The epsilon value was added to prevent any value error messages about division by zero.


The second part of the assignment was to build our own multi-layer perceptron model.  The functions used to complete this goal are:

def _initialize(self, X, y): 
This performs one-hot encoding for the target class values and randomly sets the value of the neural network weights and biases.

def fit(self, X, y): 
This method is used to train the model using gradient descent. It performs forward and backward propagations for a given number of iterations. The loss is calculated and recorded every 20 iterations as instructed. The weights and biases are updated using gradient descent.

def predict(self, X): 
This predicts the class values for new data using the already fit classifier model. It performs a forward pass through the network and returns the class with the highest probability for each test sample.