using ScikitLearn
using CSV
using DataFrames
using Statistics
using VegaLite
using XGBoost
@sk_import metrics: accuracy_score

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/X_train_enc.csv"))
df_valid = DataFrame(CSV.File("./data/X_test_enc.csv"))
X_train, X_valid = Matrix(df_train[:, 2:end]), Matrix(df_valid[:, 2:end])
y_train, y_valid = df_train[:, 1], df_valid[:, 1]

# create and train a gradient boosted tree model of 5 trees
bst = xgboost((X_train, y_train), num_round=100, max_depth=6, η=0.8, num_class= 2, objective="multi:softmax")

# predict
y_pred = XGBoost.predict(bst, X_valid)

# accuracy
accuracy_score(y_valid, y_pred) # 0.70

# # using CV
# NOTE: It looks like XGBoost.jl is currently under heavy development and nfold_cv() got lost during September this year...
# dtrain = DMatrix((X_train, y_train))
# params = Dict(
#     "max_depth"=>40, 
#     "η"=>0.8,
#     "num_class"=> 2
# )
# num_boost_round = 5
# nfold = 10 # 10-fold CV
# nfold_cv(params, dtrain, num_boost_round, nfold, seed=123) # doesn't work...

# Predict for submission
df_test = DataFrame(CSV.File("./data/df_test_enc.csv"))
X_test = Matrix(df_test)
Titanic.predict_submission(bst, X_test, "./data/submission_xgboost_withTitle.csv")

passengerid = DataFrame(CSV.File("./data/test.csv"))[:,1]
y_pred = XGBoost.predict(bst, X_test)

df_pred = DataFrame([passengerid, y_pred], ["PassengerId", "Survived"])

CSV.write("./data/submission_xgboost_withTitle.csv", df_pred)


