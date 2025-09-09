# Micrograd Info Sheet

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




