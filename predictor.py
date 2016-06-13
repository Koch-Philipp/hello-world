import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

# data (samples x features): values of features of each peptide
# labels (samples): classification (0,1) of each peptide
data, labels = load_svmlight_file("/Users/sophiamersmann/Documents/Uni/Studium/Semester/ComputationalImmunomics/Projekt/project_training_prepared.txt")

# standardization of data
scaler = MaxAbsScaler().fit(data) # MinMaxScaler, RobustScaler, StandardScaler
scaled_data = scaler.transform(data)

# normalization of data # TODO slightly better results witout normalization
normalizer = Normalizer().fit(data)
normalized_data = normalizer.transform(scaled_data)

# final estimator
svc = SVC(class_weight='balanced')

# pipeline of transforms with a final estimator
pipeline = Pipeline([('scaler', scaler), ('normalizer', normalizer), ('svc', svc)])

# stratified k-fold cross-validation
cv = StratifiedKFold(labels, n_folds=5, shuffle=True)

# print scaler.get_params()
# print normalizer.get_params()
# print svc.get_params()

C_range = np.logspace(start=-10, stop=10, base=10)
gamma_range = np.logspace(start=-10, stop=10, base=10)

# parameters of grid search
# param_grid = [
#   #{'normalizer__norm': ['l1', 'l2', 'max'], 'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear']},
#   {'normalizer__norm': ['l1', 'l2', 'max'], 'svc__C': C_range, 'svc__gamma': gamma_range, 'svc__kernel': ['rbf']}
#   #{'normalizer__norm': ['l1', 'l2', 'max'], 'svc__C': [1, 10, 100, 1000], 'svc__gamma': [0.001, 0.0001], 'svc__degree': [1,2,3,4,5,6], 'svc__coef0': [0.001, 0.0001], 'svc__kernel': ['poly']}
# ]

# parameter distributions of grid search

# print scaler.get_params().keys()
# print normalizer.get_params().keys()
# print svc.get_params().keys()

# print pipeline.get_params().keys()


# specify and run randomized grid search
grid_search = RandomizedSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=cv, error_score=np.NaN, verbose=100)
grid_search.fit(data, labels)

print grid_search.best_estimator_
print grid_search.best_score_
print grid_search.best_params_
print grid_search.scorer_
