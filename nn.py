import random
from engine import Value  # Importing the Value class for creating scalar values with automatic differentiation

class Module:
    """Base class representing a module in a neural network."""

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Returns the list of parameters of the module."""
        return []

class Neuron(Module):
    """Represents a single neuron in a neural network."""

    def __init__(self, nin, nonlin=True):
        """
        Initializes a neuron with 'nin' input connections.
        
        Args:
            nin (int): Number of input connections (input size).
            nonlin (bool): If True, applies a ReLU non-linearity to the output.
        """
        # Initialize weights with random values between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)  # Bias initialized to 0
        self.nonlin = nonlin  # Flag indicating whether to use a non-linearity

    def __call__(self, x):
        """
        Forward pass through the neuron.
        
        Args:
            x (list[Value]): Input values to the neuron.
        
        Returns:
            Value: Output of the neuron after applying the optional non-linearity.
        """
        # Compute the weighted sum of inputs plus the bias
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Apply ReLU if nonlin is True, otherwise return the linear activation
        return act.relu() if self.nonlin else act

    def parameters(self):
        """Returns the list of parameters (weights and bias) of the neuron."""
        return self.w + [self.b]

    def __repr__(self):
        """String representation of the neuron."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """Represents a layer consisting of multiple neurons."""

    def __init__(self, nin, nout, **kwargs):
        """
        Initializes a layer with 'nout' neurons, each having 'nin' input connections.
        
        Args:
            nin (int): Number of input connections (input size for each neuron).
            nout (int): Number of neurons in the layer.
        """
        # Create a list of Neuron objects
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x (list[Value]): Input values to the layer.
        
        Returns:
            list[Value] or Value: Output of the layer. If there is only one neuron, returns a single value.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Returns the list of parameters of all neurons in the layer."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """String representation of the layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """Represents a Multi-Layer Perceptron (MLP) consisting of multiple layers."""

    def __init__(self, nin, nouts):
        """
        Initializes an MLP with specified layer sizes.
        
        Args:
            nin (int): Number of input connections to the first layer.
            nouts (list[int]): List of output sizes for each layer, defining the network architecture.
        """
        # Create layers based on input size and the list of output sizes
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Forward pass through the entire MLP.
        
        Args:
            x (list[Value]): Input values to the MLP.
        
        Returns:
            Value or list[Value]: Output of the final layer of the MLP.
        """
        for layer in self.layers:
            x = layer(x)  # Pass the input through each layer sequentially
        return x

    def parameters(self):
        """Returns the list of parameters of all layers in the MLP."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """String representation of the MLP."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

