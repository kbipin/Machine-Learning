# decision tree for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)

score = []
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    score.append([i,v])
    
sorted_score = sorted(score, key = lambda score: score[1], reverse=True)

for i in range(0, len(sorted_score)):
    print('Feature:, Score: %.5f', sorted_score[i])
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()