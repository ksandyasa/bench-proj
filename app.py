from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def run_demo():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    print("Model trained. Classes:", clf.classes_)

if __name__ == "__main__":
    run_demo()