import torch
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
        assert isinstance(other, (int, float))
        other = other if isinstance(other, Value) else Value(other)
        x = self.value
        out = Value(value=self.value**other, _children=(self,), _op='pow')
        def _backward():
            self.grad += (other * self.value ** (other-1)) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append([v._label, v])
        build_topo(self)
        topo = sorted(topo, key=lambda x: x[0])
        self.grad = 1.0
        for node in reversed(topo):
            node[1]._backward()
        #printing grads
        for node in topo:
            print(node[1].grad)

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

class Layer:
    def __init__(self, input_neurons, output_neurons):
        # creating a list of neurons in the layer by creating
        # x number of input neurons and
        self.neurons = [Neuron(input_neurons) for _ in range(output_neurons)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x