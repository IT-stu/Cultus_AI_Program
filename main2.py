
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
X, y = make_blobs(
    n_samples=500,
    centers=2,
    n_features=2,
    cluster_std=2.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = len(y)
    epsilon = 1e-15
    return - (1 / m) * np.sum(
        y * np.log(y_hat + epsilon) +
        (1 - y) * np.log(1 - y_hat + epsilon)
    )


def train_logistic_regression(X, y, lr=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []

    for _ in range(iterations):
        z = np.dot(X, weights) + bias
        y_hat = sigmoid(z)

        loss = compute_loss(y, y_hat)
        losses.append(loss)

        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias, losses


weights, bias, losses = train_logistic_regression(X_train, y_train)
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_hat = sigmoid(z)
    return (y_hat >= 0.5).astype(int)


y_pred = predict(X_test, weights, bias)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))


plt.figure()

plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="Class 0")
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="Class 1")

x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals = -(weights[0] * x1_vals + bias) / weights[1]

plt.plot(x1_vals, x2_vals, label="Decision Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()
