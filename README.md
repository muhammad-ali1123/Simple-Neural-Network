# Micrograd Info Sheet

# Micrograd: A Minimal Neural Network Implementation

## Overview

Micrograd is a lightweight implementation of a neural network from scratch, featuring automatic differentiation (backpropagation) and a simple multi-layer perceptron (MLP). This implementation demonstrates the core concepts of neural networks and gradient-based optimization using only Python's standard library and basic mathematical operations.

## Key Features

- **Automatic Differentiation**: Custom `Value` class that tracks gradients through computational graphs
- **Neural Network Architecture**: Modular design with Neuron, Layer, and MLP classes
- **Backpropagation**: Automatic gradient computation using the chain rule
- **Visualization**: Computational graph visualization using Graphviz
- **41 Parameter Network**: Creates a 3-4-4-1 architecture with 41 trainable parameters

## Core Components

### 1. Value Class

The `Value` class is the foundation of the automatic differentiation system. It wraps scalar values and tracks their gradients through operations.

**Key Features:**
- Stores data and gradient values
- Tracks computational graph through `_prev` (children) and `_op` (operation)
- Implements basic arithmetic operations (`+`, `-`, `*`, `/`, `**`)
- Supports activation functions (`tanh`, `exp`)
- Automatic gradient computation via `backward()` method

**Example Usage:**
```python
a = Value(2.0)
b = Value(4.0)
c = a + b  # c.data = 6.0
c.backward()  # Computes gradients: a.grad = 1.0, b.grad = 1.0
```

### 2. Neural Network Architecture

#### Neuron Class
- **Input**: Number of input features (`nin`)
- **Parameters**: Random weights (`self.w`) and bias (`self.b`)
- **Activation**: Hyperbolic tangent (tanh)
- **Output**: Single scalar value

#### Layer Class
- **Input**: Number of inputs (`nin`) and outputs (`nout`)
- **Structure**: Collection of `nout` neurons
- **Output**: List of neuron outputs (or single value if `nout=1`)

#### MLP (Multi-Layer Perceptron) Class
- **Input**: Input size (`nin`) and list of layer sizes (`nouts`)
- **Architecture**: Sequential layers with tanh activations
- **Forward Pass**: Data flows through all layers sequentially

### 3. Network Architecture Details

The implemented network has the following structure:
```
Input Layer:  3 neurons (features)
Hidden Layer 1: 4 neurons (3×4 weights + 4 biases = 16 parameters)
Hidden Layer 2: 4 neurons (4×4 weights + 4 biases = 20 parameters)
Output Layer:  1 neuron  (4×1 weights + 1 bias = 5 parameters)
Total Parameters: 16 + 20 + 5 = 41 parameters
```

## Training Process

### Dataset
```python
xs = [
    [2.0, 3.0, -1.0],   # Input sample 1
    [3.0, -1.0, 0.5],   # Input sample 2
    [0.5, 1.0, 1.0],    # Input sample 3
    [1.0, 1.0, -1.0],   # Input sample 4
]
ys = [1.0, -1.0, -1.0, 1.0]  # Target outputs
```

### Training Loop
1. **Forward Pass**: Compute predictions for all samples
2. **Loss Calculation**: Mean squared error between predictions and targets
3. **Backward Pass**: Compute gradients using automatic differentiation
4. **Parameter Update**: Gradient descent with learning rate 0.1
5. **Gradient Reset**: Zero out gradients for next iteration

### Loss Function
```python
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
```

## Usage Example

```python
# Create network
n = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4, 1 output

# Single prediction
x = [2.0, 3.0, -1.0]
output = n(x)  # Returns Value object

# Training
for epoch in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
    
    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    for p in n.parameters():
        p.data += -0.1 * p.grad
    
    print(f"Epoch {epoch}: Loss = {loss.data}")
```

## Visualization

The implementation includes computational graph visualization using Graphviz:

```python
# Visualize a simple computation
x = Value(2.0, label='x')
y = x.tanh()
y.backward()
draw_dot(y)  # Shows computational graph with gradients
```

## Training Results

The network successfully learns the given pattern, with loss decreasing from ~6.65 to ~2.39 over 20 epochs:

```
Epoch 0: Loss = 6.653409497357662
Epoch 5: Loss = 5.264217138504994
Epoch 10: Loss = 3.756007237225296
Epoch 19: Loss = 2.3906082754386144
```

## Implementation Notes

### Automatic Differentiation
- Uses reverse-mode automatic differentiation (backpropagation)
- Builds computational graph during forward pass
- Computes gradients in reverse topological order

### Numerical Considerations
- All computations use Python floats (double precision)
- Gradients are accumulated additively
- Manual gradient zeroing required between iterations

### Comparison with PyTorch
The implementation produces results consistent with PyTorch, demonstrating correctness of the gradient computation:

```python
# PyTorch comparison shows matching gradients
x1.grad.item() # -1.5000003851533106 (PyTorch)
# vs micrograd equivalent computation
```

## Limitations

1. **Scalability**: Not optimized for large networks or datasets
2. **Operations**: Limited set of mathematical operations
3. **Activations**: Only tanh and exp implemented
4. **Optimization**: Only basic gradient descent (no momentum, Adam, etc.)
5. **Data Types**: Only supports scalar operations (no tensors/matrices)

## Extensions

Potential improvements and extensions:
- Additional activation functions (ReLU, sigmoid, etc.)
- More optimization algorithms
- Regularization techniques
- Batch processing
- Convolutional layers
- Better numerical stability

## Dependencies

- Python 3.x
- `graphviz` (for visualization)
- `matplotlib` (for plotting)
- `numpy` (optional, for comparisons)

This micrograd implementation serves as an excellent educational tool for understanding the fundamentals of neural networks, automatic differentiation, and gradient-based optimization from first principles.




### What are Neural Networks?
Neural networks are mathematical models inspired by how the human brain works. They take inputs (data), multiply them by weights (which reflect importance), add a bias, and pass the result through an activation function to produce an output.

This process is called a **forward pass**. The output is compared with the actual answer using a loss function, which measures how wrong the prediction was. A smaller loss means the model is more accurate.

To improve, the network uses **backward propagation**: weights and biases are adjusted so that the loss decreases. This process repeats many times until the network becomes accurate at solving the problem.

*For example, the ChatGPT model has billions of texts from the internet to be used as parameters to improve its accuracy and create accurate predictions.*


### Scale of Neural Networks
The neural network I built has just 41 parameters, but modern networks can have billions of parameters.
* Small networks can solve toy problems.
* Large models like GPT are trained on vast amounts of text and use billions of parameters to predict the next word in a sequence.
* Despite the scale difference, both small and large networks operate on the same fundamental principles.

### How a Node Works (Step-by-Step)
Each node contains:
* Inputs ($X$)
* Weights ($W$)
* A bias ($b$)
* An output ($y$)

**Process:**
1.  Multiply each input by its weight.
2.  Add them together.
3.  Add the bias.
4.  Pass the result through an activation function.

If the result is above the threshold, the node “fires” and passes data forward.

### Example: Should I Go to Wonderland Today?
* **Inputs (X):**
    * $X_1 = 1$: It’s not raining today.
    * $X_2 = 0$: It’s busy today.
    * $X_3 = 1$: My friends are excited to go.
* **Weights (W):**
    * $W_1 = 5$: No rain → big positive factor.
    * $W_2 = 2$: Crowds don’t matter much.
    * $W_3 = 4$: I don’t want to go alone.
* **Bias:** $-3$ (threshold = 3)

**Computation:**
$f(x) = 1 \times 5 + 0 \times 2 + 1 \times 4 - 3 = 6$

**Decision rule:**
* If $f(x) > 0$, output = 1 (Yes).
* If $f(x) \le 0$, output = 0 (No).

Since $6>0$, the final decision is Yes → We’re going to Wonderland!


### Importance of Backward Propagation
The **loss function** measures how wrong a neural network is:
Prediction ($\hat{y}$) vs. True answer ($y$).
* **Example:** if $y=1$ but $\hat{y}=0.1$, the loss is large.
* If $y=1$ and $\hat{y}=0.95$, the loss is small.

The goal is always to minimize loss, which makes predictions more accurate. This is done by updating weights and biases.


### All in all…
* **Forward propagation** = Make a prediction.
* **Loss function** = Measure how wrong it was.
* **Backward propagation** = Adjust weights to improve.

This cycle repeats billions of times until the network is accurate.

### In Conclusion…
A neural network is a machine learning system that makes decisions by:
* Identifying input signals.
* Weighing their importance.
* Producing an output.

Most neural networks have:
* **Input layer** → receives data.
* **Hidden layer(s)** → process information.
* **Output layer** → produces the final prediction.

If the output of a node is greater than a certain threshold (bias), it activates and passes information forward. Otherwise, it stays inactive. Over time, using training data, the network adjusts itself to improve accuracy.




