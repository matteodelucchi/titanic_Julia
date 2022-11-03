using ScikitLearn
using CSV
using DataFrames
using Statistics
using VegaLite
using XGBoost: randomforest
# @sk_import model_selection: learning_curve
# @sk_import inspection: permutation_importance
# @sk_import metrics: mean_squared_error
@sk_import metrics: accuracy_score
# @sk_import model_selection: train_test_split
# @sk_import model_selection: GridSearchCV
# @sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/X_train_enc.csv"))
df_valid = DataFrame(CSV.File("./data/X_test_enc.csv"))
X_train, X_valid = Matrix(df_train[:, 2:end]), Matrix(df_valid[:, 2:end])
# X_train, X_valid = df_train[:, 2:end], df_valid[:, 2:end]
y_train, y_valid = df_train[:, 1], df_valid[:, 1]

# create and train a gradient boosted tree model of 5 trees
bst = xgboost((X_train, y_train), num_round=100, max_depth=40, η=0.8, num_class= 2, objective="multi:softmax")

# predict
y_pred = predict(bst, X_valid)

# accuracy
accuracy_score(y_valid, y_pred)
