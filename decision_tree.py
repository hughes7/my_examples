from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load iris datasets
dataset = datasets.load_iris()

# fit a model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
