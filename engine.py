class Value:
    """ stores a single scalar value and its gradient """
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data  # The actual scalar value this node holds
        self.grad = 0  # Gradient initialized to zero for backpropagation
        # Internal variables for constructing the computational graph
        self._backward = lambda: None  # Function to compute the local gradient during backpropagation
        self._prev = set(_children)  # Set of parent nodes (dependencies)
        self._op = _op  # Operation that created this node, useful for visualization/debugging
    
    def __add__(self, other):
        # Ensure 'other' is a Value object for consistent operations
        other = other if isinstance(other, Value) else Value(other)
        # Create a new Value representing the result of addition
        out = Value(self.data + other.data, (self, other), '+')
        
        # Define the backward function for computing gradients for addition
        def _backward():
            self.grad += out.grad  # Gradient contribution to 'self'
            other.grad += out.grad  # Gradient contribution to 'other'
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        # Ensure 'other' is a Value object for consistent operations
        other = other if isinstance(other, Value) else Value(other)
        # Create a new Value representing the result of multiplication
        out = Value(self.data * other.data, (self, other), '*')
        
        # Define the backward function for computing gradients for multiplication
        def _backward():
            self.grad += other.data * out.grad  # Chain rule: derivative w.r.t 'self'
            other.grad += self.data * out.grad  # Chain rule: derivative w.r.t 'other'
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        # Ensure the power is an int or float
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # Create a new Value representing the result of raising 'self' to the power 'other'
        out = Value(self.data**other, (self,), f'**{other}')
        
        # Define the backward function for computing gradients for power operation
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad  # Chain rule for power
        out._backward = _backward

        return out
    
    def relu(self):
        # ReLU operation: max(0, self.data)
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        # Define the backward function for computing gradients for ReLU
        def _backward():
            self.grad += (out.data > 0) * out.grad  # Gradient only passes if the output is positive
        out._backward = _backward

        return out
    
    def backward(self):
        # Topologically order the nodes to propagate gradients in the correct sequence
        topo = []
        visited = set()
        
        # Recursive function to build the topological ordering
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Start with the gradient of the final output node set to 1 (chain rule)
        self.grad = 1
        # Apply the chain rule in reverse topological order
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self):  # Negation (-self)
        return self * -1

    def __radd__(self, other):  # For expressions like (other + self)
        return self + other

    def __sub__(self, other):  # Subtraction (self - other)
        return self + (-other)

    def __rsub__(self, other):  # For expressions like (other - self)
        return other + (-self)

    def __rmul__(self, other):  # For expressions like (other * self)
        return self * other

    def __truediv__(self, other):  # Division (self / other)
        return self * other**-1

    def __rtruediv__(self, other):  # For expressions like (other / self)
        return other * self**-1

    def __repr__(self):
        # String representation of the Value object showing data and gradient
        return f"Value(data={self.data}, grad={self.grad})"
