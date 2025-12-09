from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logreg(X_train, y_train):
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    return logreg


def train_random_forest(X_train, y_train, n_estimators: int = 200, random_state: int = 42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf
