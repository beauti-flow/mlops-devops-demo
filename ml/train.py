# ml/train.py
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib
##
def train_and_save_model():
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()
