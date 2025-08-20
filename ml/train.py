
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

"""
# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("ml/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")

"""