import random
from engine import Value  # Import the Value class for autograd
from nn import MLP  # Import the MLP class to create the model

# Create a simple dataset (e.g., XOR problem)
data = [
    ([2.0, 3.0], 1.0),   # Input: [2.0, 3.0], Expected output: 1.0
    ([3.0, -1.0], -1.0), # Input: [3.0, -1.0], Expected output: -1.0
    ([1.0, 1.0], 1.0),   # Input: [1.0, 1.0], Expected output: 1.0
    ([2.0, -2.0], -1.0)  # Input: [2.0, -2.0], Expected output: -1.0
]

# Initialize a simple MLP model: 2 inputs, one hidden layer with 4 neurons, 1 output
model = MLP(2, [4, 1])  # MLP architecture: input layer with 2 nodes, hidden layer with 4 neurons, output layer with 1 neuron

# Training loop
epochs = 100  # Number of training iterations
learning_rate = 0.01  # Learning rate for gradient descent

for k in range(epochs):
    # Forward pass: predict the output for each data point
    total_loss = Value(0)  # Initialize total loss for this epoch
    for x, y in data:
        x = [Value(xi) for xi in x]  # Convert each input value to a Value object for autograd tracking
        y_pred = model(x)  # Forward pass through the model to get predictions
        loss = (y_pred - Value(y)) ** 2  # Calculate mean squared error loss
        total_loss += loss  # Accumulate the loss for each data point
    
    # Backward pass: reset gradients, calculate new gradients
    model.zero_grad()  # Clear previous gradients to avoid accumulation
    total_loss.backward()  # Perform backpropagation to compute gradients
    
    # Update model parameters using gradient descent
    for p in model.parameters():
        p.data -= learning_rate * p.grad  # Update parameters using gradient descent rule
    
    # Print the progress: epoch number and current loss
    print(k, total_loss.data)  # Output the epoch index and the loss value
