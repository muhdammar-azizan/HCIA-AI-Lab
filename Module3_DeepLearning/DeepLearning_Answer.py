"""
Module 3 - Deep Learning Lab Guide
Questions
"""

import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist
from util import smooth_curve
from multi_layer_net import MultiLayerNet
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# ============================================================
# Question 1: How to change the learning rate in the code?
# Use different learning rate setting policies and observe the loss curve.
# ============================================================

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1000

# Test three different learning rate values using the SGD optimizer.
optimizers_q1 = {}
optimizers_q1['SGD (lr=0.01)'] = SGD(lr=0.01)
optimizers_q1['SGD (lr=0.1)'] = SGD(lr=0.1)
optimizers_q1['SGD (lr=1.0)'] = SGD(lr=1.0)

networks_q1 = {}
train_loss_q1 = {}
for key in optimizers_q1.keys():
    networks_q1[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss_q1[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers_q1.keys():
        grads = networks_q1[key].gradient(x_batch, t_batch)
        optimizers_q1[key].update(networks_q1[key].params, grads)

        loss = networks_q1[key].loss(x_batch, t_batch)
        train_loss_q1[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers_q1.keys():
            loss = networks_q1[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

markers_q1 = {"SGD (lr=0.01)": "o", "SGD (lr=0.1)": "x", "SGD (lr=1.0)": "s"}
x_axis_q1 = np.arange(max_iterations)
for key in optimizers_q1.keys():
    plt.plot(x_axis_q1, smooth_curve(train_loss_q1[key]), marker=markers_q1[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2)
plt.title("Question 1: Effect of Learning Rate on SGD")
plt.legend()
plt.show()

"""
Conclusion (Question 1):
Changing the learning rate (lr) parameter when creating an optimizer object
(e.g., SGD(lr=0.01)) directly controls the step size used to update the
model's weights during training.

Three learning rates were tested using the SGD optimizer:
- lr = 0.01: The loss decreased steadily but relatively slowly, reaching
  around 0.2-0.3 after 900 iterations.
- lr = 0.1: The loss decreased the fastest and most smoothly, reaching the
  lowest final value (around 0.03-0.08). This learning rate provided the
  best balance between speed and stability for this network.
- lr = 1.0: The loss curve was highly unstable, fluctuating sharply and
  sometimes increasing again rather than steadily decreasing. This shows
  that a learning rate that is too large causes the optimizer to overshoot
  the optimal weights repeatedly, preventing stable convergence.

This confirms that the learning rate must be carefully tuned: too small
leads to slow learning, while too large leads to unstable or divergent
training.

"""

# ============================================================
# Question 2: Add a test module to the code to generate the
# test accuracy in different phases.
# ============================================================

optimizers_q2 = {}
optimizers_q2['SGD'] = SGD()
optimizers_q2['Momentum'] = Momentum()
optimizers_q2['AdaGrad'] = AdaGrad()
optimizers_q2['Adam'] = Adam()

networks_q2 = {}
train_loss_q2 = {}
train_acc_q2 = {}
test_acc_q2 = {}
for key in optimizers_q2.keys():
    networks_q2[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss_q2[key] = []
    train_acc_q2[key] = []
    test_acc_q2[key] = []

iter_per_epoch = int(max(train_size / batch_size, 1))

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers_q2.keys():
        grads = networks_q2[key].gradient(x_batch, t_batch)
        optimizers_q2[key].update(networks_q2[key].params, grads)

        loss = networks_q2[key].loss(x_batch, t_batch)
        train_loss_q2[key].append(loss)

    # Test module: evaluate train and test accuracy once per epoch.
    if i % iter_per_epoch == 0:
        print("===========" + "epoch (iteration " + str(i) + ")" + "===========")
        for key in optimizers_q2.keys():
            train_acc = networks_q2[key].accuracy(x_train, t_train)
            test_acc = networks_q2[key].accuracy(x_test, t_test)
            train_acc_q2[key].append(train_acc)
            test_acc_q2[key].append(test_acc)
            print(key + " -> train acc: " + str(train_acc) + ", test acc: " + str(test_acc))

# Visualize how test accuracy improves across epochs for each optimizer.
markers_q2 = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
for key in optimizers_q2.keys():
    x_axis_q2 = np.arange(len(test_acc_q2[key]))
    plt.plot(x_axis_q2, test_acc_q2[key], marker=markers_q2[key], label=key)
plt.xlabel("epochs")
plt.ylabel("test accuracy")
plt.ylim(0, 1.0)
plt.title("Question 2: Test Accuracy Across Training Phases")
plt.legend()
plt.show()

"""
Conclusion (Question 2):
A test module was added to evaluate both training accuracy and test
accuracy at the start of every epoch (using x_test, t_test which were
loaded but unused in the original code). This shows how well the model
generalizes to unseen data, not just how well it memorizes training data.

Results across 3 phases (epochs 0, 1, 2) showed:
- Epoch 0: All optimizers started near random-guess accuracy (~10-17%).
- Epoch 1: A large jump to 88-96% accuracy, showing the network quickly
  learned the basic patterns in the data.
- Epoch 2: A smaller further improvement to 91-97%, as the network
  continued to refine its weights.

SGD consistently showed slightly lower accuracy than Momentum, AdaGrad,
and Adam at every phase, confirming that more advanced optimizers converge
faster and reach higher accuracy within the same number of iterations.

"""