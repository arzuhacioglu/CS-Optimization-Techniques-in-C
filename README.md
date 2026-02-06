# CS Optimization Techniques in C

> **Technical Showcase:** A low-level implementation of Neural Network optimization algorithms (GD, SGD, ADAM) built from scratch in C without external ML libraries.

## Project Overview

This project explores the mathematical engine behind Deep Learning. Instead of relying on high-level frameworks like TensorFlow or PyTorch, I implemented a **Single-Layer Neural Network** purely in C to solve a binary classification problem (MNIST Digits: 4 vs 8).

The primary goal was not just to build a classifier, but to perform a comparative study of how different **mathematical optimization techniques** navigate the loss surface and converge to a global minimum using manual calculus.

### Technical Implementation Details
* **Language:** C (Standard Library)
* **Architecture:** Single Perceptron (784 Input Neurons + 1 Bias $\rightarrow$ 1 Output).
* **Activation Function:** Hyperbolic Tangent (`tanh`) for non-linearity.
* **Loss Function:** Mean Squared Error (MSE) calculated manually.
* **Differentiation:** Gradients ($\nabla$) are computed via **Backpropagation** using the Chain Rule.

---

## Algorithms Implemented & Analyzed

I engineered three distinct optimization strategies to train the network:

### 1. Gradient Descent (GD)
* **Mechanism:** Updates weights using the average gradient calculated from the **entire dataset** in each iteration.
* **Behavior:** Deterministic and stable. It follows a direct, linear path towards the minimum but is computationally expensive per epoch.

### 2. Stochastic Gradient Descent (SGD)
* **Mechanism:** Updates weights using a **single random sample** per iteration.
* **Behavior:** Introduces stochastic noise (high variance). While the loss fluctuates significantly ("zig-zag" movement), this noise helps the model escape local minima and converge faster in terms of computation time.

### 3. ADAM (Adaptive Moment Estimation)
* **Mechanism:** A sophisticated optimizer that combines **Momentum** (First Moment - $m_t$) and **RMSProp** (Second Moment - $v_t$).
* **Behavior:** Adapts the learning rate for each parameter individually. In my experiments, this provided the most efficient convergence and sharpest classification boundaries.

---

## Key Achievements & Experimental Results

Based on the training data and t-SNE visualizations conducted during the research:

### 1. Convergence Performance
| Algorithm | Convergence Speed | Stability | Result Analysis |
| :--- | :---: | :---: | :--- |
| **Gradient Descent** | Slow | High | Showed a linear decrease in loss. While stable, it required more epochs to reach the same accuracy as ADAM. |
| **SGD** | Fast | Low | Successfully minimized loss but exhibited significant oscillation around the minimum due to its probabilistic nature. |
| **ADAM** | **Very Fast** | **High** | **Best Performer.** It rapidly reduced the loss to near-zero values and stabilized quickly, proving the effectiveness of adaptive learning rates. |

### 2. Cluster Separation (t-SNE)
The model successfully learned to distinguish between the digits '4' and '8'.
* **Initial State:** The data points for '4' and '8' were randomly scattered and overlapping.
* **Final State:** After training with ADAM, the t-SNE projections showed **two distinct, well-separated clusters**, proving that the custom C-based neural network effectively learned the vector representations of the handwritten digits.

###  3. Mathematical Robustness
The project demonstrated that a manual implementation of partial derivatives and matrix operations in C can achieve high-precision results (Loss approaching 0.0) comparable to high-level Python libraries, validating the mathematical formulas derived for Backpropagation.

---

##  How to Run
1.  **Clone the repository.**
2.  **Download Data:** Place the MNIST `train_images.idx` and `test_images.idx` files in the root directory.
3.  **Compile:**
    ```bash
    gcc src/mnist_optimizer.c -o optimizer_run -lm
    ```
4.  **Execute:**
    ```bash
    ./optimizer_run
    ```

---
