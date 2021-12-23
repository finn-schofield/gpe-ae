from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def eval_embedding(data, labels):
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, data, labels, cv=10)
    return scores.mean()
