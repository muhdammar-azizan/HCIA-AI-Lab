"""
Module 4 - AI Development Framework (PyTorch)
Questions
"""

# ============================================================
# Question 1: Why does the loss not decrease but increase
# during LeNet and ResNet training?
# ============================================================
"""
The loss increases or becomes unstable mainly when the learning rate is too high.
This was directly observed in Module 3's learning rate experiment: with the same
MNIST dataset, SGD with lr=0.01 and lr=0.1 decreased smoothly, but lr=1.0 caused
the loss to fluctuate sharply and even become 'nan' at times.

When the learning rate is too large, each weight update step overshoots the
optimal point on the loss surface, causing the model to jump to a worse position
instead of converging toward the minimum. In extreme cases, the weight values
grow so large that calculations overflow, producing 'nan' loss values.

Other possible causes include: poor weight initialization, exploding gradients
in very deep networks (relevant to ResNet-50's 50 layers), or a batch size that
is too small, causing noisy/unstable gradient estimates.
"""

# ============================================================
# Question 2: Can accuracy be further improved by increasing
# the number of training epochs during CIFAR-10 training of LeNet?
# ============================================================
"""
Increasing epochs can improve accuracy only up to a certain point. In early
epochs, the model is still learning useful patterns, so more epochs generally
help. However, after the model has learned most of the useful patterns in the
training data, continuing to train for more epochs leads to overfitting: the
model starts memorizing the training data (including its noise) instead of
learning patterns that generalize to new data.

This means training accuracy keeps rising, but test/validation accuracy
plateaus or even decreases. Therefore, simply adding more epochs is not a
reliable way to keep improving real-world accuracy; it should be combined with
monitoring validation performance and stopping when it stops improving (early
stopping).
"""

# ============================================================
# Question 3: How can we further improve network accuracy
# without changing the network structure?
# ============================================================
"""
Several approaches can improve accuracy without altering the network architecture:

1. Tune the learning rate (and use a learning rate scheduler, as used in the
   ResNet-50 experiment with ReduceLROnPlateau) to help the model converge
   more effectively.
2. Use a better optimizer (e.g., Adam or AdaGrad instead of plain SGD), as
   shown in Module 3 where AdaGrad and Adam converged faster than SGD.
3. Apply data augmentation (e.g., RandomHorizontalFlip, as used in the
   ResNet-50 experiment) to artificially increase the diversity of training
   data and reduce overfitting.
4. Normalize/standardize input data properly (as done with image
   normalization in LeNet and ResNet-50).
5. Train for an appropriate number of epochs while monitoring validation
   accuracy to avoid overfitting (early stopping).
6. Use regularization techniques such as dropout (already used in TextCNN)
   or weight decay (already used in ResNet-50's optimizer).
7. Increase the amount and quality of training data, if available.
"""

print("Module 4 Questions answered - see docstrings above.")