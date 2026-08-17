
import random
import math
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        dot.node(name=str(id(n)), label="{ %s | data %.4f | grad %.4f }" % (n._label, n.value, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


class Value():
    def __init__(self, value, _children=(), _op='', _label=''):
        self.value = value
        self.children = _children
        self.grad = 0.0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None
        self._label = _label
    def __repr__(self):
        return f"Value(data={self.value})"
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __neg__(self):
        return self * -1
    def __sub__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    # handles reversed multiplication
    def __radd__(self,other):
        return self + other
    def __rmul__(self,other):
        return self * other
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out
    def __truediv__(self,other):
        return self * other**-1
    def __rtruediv__(self,other):
        return other * self**-1
    def __floordiv__(self,other):
        return Value(self.value // other.value, (self, other), '//')
    def tanh(self):
        x = self.value
        # If x is large and positive, e^(-2x) approaches 0, which is perfectly stable
        if x >= 0:
            t = (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x))
        else:
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(value=t, _children=(self,), _op='tanh')
        def _backward():
            #this is because local derivative of tanh function is (1-tanh^2)
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x = self.value
        out = Value(value=math.exp(x), _children=(self,), _op='exp')
        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward
        return out
    def __pow__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(value=self.value**other.value, _children=(self, other), _op='pow')
        def _backward():
            self.grad += (other.value * self.value ** (other.value-1)) * out.grad
            if self.value > 0:
                other.grad += (out.value * math.log(self.value)) * out.grad
        out._backward = _backward
        return out
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other ** self
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
class Neuron:
    def __init__(self, input_neurons):
        # setting w as a list of random weights for the neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range(input_neurons)]
        # b is just that exact neuron's bias
        self.b = Value(random.uniform(-1,1))
        # call attribute ensures that when this class is called, the neuron's output data is
        # generated through a sum of the products of its weights and bias and then
        # and then its passed through a squashing function tanh
    def __call__(self, x):
        # this is essentially (weights * x) + bias where x is the input data from the previous neuron/s
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
        

class Layer:
    def __init__(self, input_neurons, output_neurons):
        # creating a list of neurons in the layer by creating
        # x number of input neurons and
        self.neurons = [Neuron(input_neurons) for _ in range(output_neurons)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def save(self, filename):
        with open(filename, 'w') as fob:
            fob.write("value,grad\n")
            for p in self.parameters():
                fob.write(f"{p.value},{p.grad}\n")
    def load(self, filename):
        try:
            with open(filename, 'r') as fob:
                lines = fob.readlines()[1:] # skip header
        except FileNotFoundError:
            print(f"File {filename} not found, proceeding with randomly initialized weights.")
            return

        params = self.parameters()
        if len(lines) != len(params):
            print(f"Warning: Model has {len(params)} params but file has {len(lines)}.")

        for p, line in zip(params, lines):
            try:
                val, grad = line.strip().split(',')
                p.value = float(val)
                p.grad = float(grad)
            except ValueError:
                pass
    

# Generate a dataset with a hidden non-linear relation
# Relation: y = 1.0 if the point is inside a sphere of radius 1.5, else -1.0
random.seed(42)
xs = []
ys = []
for _ in range(30):
    x1, x2, x3 = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)
    xs.append([x1, x2, x3])
    ys.append(1.0 if x1**2 + x2**2 + x3**2 < 1.5**2 else -1.0)

n = MLP(3, [4, 4, 1])
n.load('model_parameters.csv')

def retrain():
    epochs = 5000
    learning_rate = 0.05
    print("\nStarting training...")
    for k in range(epochs):
        ypred = [n(x)[0] for x in xs]
        
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        for p in n.parameters():
            p.grad = 0.0
            
        loss.backward()
        
        for p in n.parameters():
            p.value += -learning_rate * p.grad
            
        if k % 50 == 0:
            print(f"Epoch {k} | Loss: {loss.value:.4f}")
    
    n.save('model_parameters.csv')
    print("Training complete. Model saved.")

def test():
    print("\n--- Testing Mode ---")
    while True:
        try:
            a_str = input("Enter input a (or type 'q' to quit): ")
            if a_str.lower() == 'q':
                break
            a = float(a_str)
            b = float(input("Enter input b: "))
            c = float(input("Enter input c: "))
        except ValueError:
            print("Invalid input! Please enter numbers.")
            continue
            
        expected = 1.0 if a**2 + b**2 + c**2 < 1.5**2 else -1.0
        pred = n([a, b, c])[0]
        deviation = abs(pred.value - expected)
        
        print(f"\nPrediction: {pred.value:.4f}")
        print(f"Expected:   {expected:.4f}")
        print(f"Deviation:  {deviation:.4f}\n")

if __name__ == "__main__":
    while True:
        print("\n=== Main Menu ===")
        print("1. Retrain the model")
        print("2. Test the model manually")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            retrain()
        elif choice == '2':
            test()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")