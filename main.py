import math
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
a = Value(0.5, _label='a')
b = Value(0.4, _label='b')
c = Value(0.2, _label='c')
d = Value(0.3, _label='d')
e = a*a
# e._label = 'e'
# f = c+d
# f._label = 'f'
# g = e*f
# g._label = 'g'
# i = g.tanh()
"""
ANYTHING BELOW THIS IS UNNECESSARY NOW SINCE WE ARE DOING BACKPROPAGATION VIA THE INTERNAL FUNCTION backward()
"""
# print('A =', a)
# print('B =', b)
# print('C =', c)
# print('D =', d)
# print('A*B =', e)
# print('C+D =', f)
# print('(A*B) * (C+D) =', g)
# print('tanh{(A*B) * (C+D)} =', i)
# i.grad = 1.0
# i._backward()
# g._backward()
# print(g.grad)
"""
ANYTHING ABOVE THIS IS UNNECESSARY NOW SINCE WE ARE DOING BACKPROPAGATION AUTOMATICLALY VIA THE INTERNAL FUNCTION backward()
"""
e.backward()

