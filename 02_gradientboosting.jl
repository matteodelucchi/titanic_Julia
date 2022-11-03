using Titanic
using ScikitLearn
using CSV
using DataFrames
using Statistics
using VegaLite
using JLD
@sk_import model_selection: learning_curve
@sk_import inspection: permutation_importance
@sk_import metrics: mean_squared_error
@sk_import metrics: accuracy_score
@sk_import model_selection: train_test_split
@sk_import model_selection: GridSearchCV
@sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/X_train_enc.csv"))
df_valid = DataFrame(CSV.File("./data/X_test_enc.csv"))
X_train, X_valid = Matrix(df_train[:, 2:end]), Matrix(df_valid[:, 2:end])
y_train, y_valid = df_train[:, 1], df_valid[:, 1]

# set learning parameters
# params = Dict(
#     "n_estimators"=> [50, 60, 70, 80, 100, 200,300,400,500],
#     "max_depth"=> [4,5,6,7,8],
#     "min_samples_split"=> [3,5,8,10, 12, 13],
#     "learning_rate"=> [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
#     "loss"=> ["log_loss", "exponential"],
#     "random_state"=> [18]
# )
params = Dict(
    "n_estimators"=> [100],
    "max_depth"=> [6],
    "min_samples_split"=> [10],
    "learning_rate"=> [0.1],
    "loss"=> ["exponential"],
    "random_state"=> [18]
)

# Fit a classifier
gbcl_base_model = GradientBoostingClassifier()
# instantiate a CV scheme
clf_model = GridSearchCV(gbcl_base_model, params, n_jobs=5)
# Fit classifier using CV scheme
@time clf_model_fit = clf_model.fit(X_train, y_train)

# Extract best params, fit model with best params and make prediction
best_params = clf_model_fit.best_params_
clf_model_fit_best = GradientBoostingClassifier(max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], min_samples_split=best_params["min_samples_split"], random_state=best_params["random_state"], loss=best_params["loss"]).fit(X_train, y_train)
y_pred = clf_model_fit_best.predict(X_valid)

# Calculate performance
println("The model score (mean accuracy) on validation set is: ", clf_model_fit_best.score(X_valid, y_valid)) # 0.82
mse = mean_squared_error(y_valid, y_pred)
println("The mean squared error (MSE) on validation set: ", mse)

# plot training curve
Titanic.plt_trainingcurve(best_params, clf_model_fit_best, X_valid, y_valid)

# Plot feature importance
Titanic.plt_featureimportances(clf_model_fit_best; feats = names(df_train[:,Not("Survived")]))

# Predict for submission
df_test = DataFrame(CSV.File("./data/df_test_enc.csv"))
X_test = Matrix(df_test)
Titanic.predict_submission(clf_model_fit_best, X_test)

# save("./clf_model__ohetitle.jld", "clf_model_fit_best", clf_model_fit_best, "clf_model", clf_model, "clf_model_fit", clf_model_fit)


#### without title
# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/X_train_enc.csv"))
df_train = df_train[:,Not("Name")]
df_valid = DataFrame(CSV.File("./data/X_test_enc.csv"))
df_valid = df_valid[:,Not("Name")]
X_train, X_valid = Matrix(df_train[:, 2:end]), Matrix(df_valid[:, 2:end])
y_train, y_valid = df_train[:, 1], df_valid[:, 1]

# set learning parameters
# params = Dict(
#     "n_estimators"=> [50, 60, 70, 80, 100, 200,300,400,500],
#     "max_depth"=> [4,5,6,7,8],
#     "min_samples_split"=> [3,5,8,10, 12, 13],
#     "learning_rate"=> [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
#     "loss"=> ["log_loss", "exponential"],
#     "random_state"=> [18]
# )
params = Dict(
    "n_estimators"=> [100],
    "max_depth"=> [6],
    "min_samples_split"=> [10],
    "learning_rate"=> [0.1],
    "loss"=> ["exponential"],
    "random_state"=> [18]
)

# Fit a classifier
gbcl_base_model = GradientBoostingClassifier()
# instantiate a CV scheme
clf_model = GridSearchCV(gbcl_base_model, params, n_jobs=5)
# Fit classifier using CV scheme
@time clf_model_fit = clf_model.fit(X_train, y_train)

# Extract best params, fit model with best params and make prediction
best_params = clf_model_fit.best_params_
clf_model_fit_best = GradientBoostingClassifier(max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], min_samples_split=best_params["min_samples_split"], random_state=best_params["random_state"], loss=best_params["loss"]).fit(X_train, y_train)
y_pred = clf_model_fit_best.predict(X_valid)

# Calculate performance
println("The model score (mean accuracy) on validation set is: ", clf_model_fit_best.score(X_valid, y_valid)) # 0.82
mse = mean_squared_error(y_valid, y_pred)
println("The mean squared error (MSE) on validation set: ", mse)

# plot training curve
Titanic.plt_trainingcurve(best_params, clf_model_fit_best, X_valid, y_valid)

# Plot feature importance
Titanic.plt_featureimportances(clf_model_fit_best; feats = names(df_train[:,Not("Survived")]))

# Predict for submission
df_test = DataFrame(CSV.File("./data/df_test_enc.csv"))
X_test = Matrix(df_test)
Titanic.predict_submission(clf_model_fit_best, X_test)

# save("./clf_model_notitle.jld", "clf_model_fit_best", clf_model_fit_best, "clf_model", clf_model, "clf_model_fit", clf_model_fit)

