# MNIST_Pytorch_example
Building 3 layer Neural Network from and evaluating the classifier performance for each class

1- Picking device type : CPU or GPU
2- (loading prtrained model)
  2.2 freezing parametrs. (Gradient no backward)
3- Define classifier (sequential)
  3.1 Linear transformation
  3.2 ReLU activation
  3.3 Dropout (not in this example)
  3.4 Softmax to calculate probabilities
4- Define Loss criterion
5- Define Optimizer
6- (Move the model to CPU/GPU, the difined device in 1) not in this example
7- Train model
  7.1 keep track of Number of epochs, training steps, loss validation loss
  7.2 Load iterated images , load it to the device=GPU/CPU (not in this example)
  7.3 Calculate loss
  7.4 Launch a backward pass
  7.4 incremente the counters, optimizer
8- Evaluate the classifier performane for each class
  8.2 show the graph validation loss (not in this example) and test loss (calculated but not showed)
9-Saving the model (not in this example)
