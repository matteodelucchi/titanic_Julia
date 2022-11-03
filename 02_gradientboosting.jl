using ScikitLearn
using CSV
using DataFrames
using Statistics
using VegaLite
@sk_import model_selection: learning_curve
@sk_import inspection: permutation_importance
@sk_import metrics: mean_squared_error
@sk_import metrics: accuracy_score
@sk_import model_selection: train_test_split
@sk_import model_selection: GridSearchCV
@sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/train_cleaned.csv"))
df_train = df_train[:, Not("title")]
X = Matrix(df_train[:, 2:end])
y = df_train[:, 1]

# Split train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=18
)

# set learning parameters
params = Dict(
    "n_estimators"=> [50, 60, 70, 80, 100, 200,300,400,500],
    "max_depth"=> [4,5,6,7,8],
    "min_samples_split"=> [3,5,8,10, 12, 13],
    "learning_rate"=> [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
    "loss"=> ["log_loss", "exponential"],
    "random_state"=> [18]
)

# Fit a classifier
gbcl_base_model = GradientBoostingClassifier()
# instantiate a CV scheme
clf_model = GridSearchCV(gbcl_base_model, params)
# Fit classifier using CV scheme
@time clf_model_fit = clf_model.fit(X_train, y_train)

# Extract best params, fit model with best params and make prediction
best_params = clf_model_fit.best_params_
clf_model_fit_best = GradientBoostingClassifier(max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], min_samples_split=best_params["min_samples_split"], random_state=best_params["random_state"]).fit(X_train, y_train)
y_pred = clf_model_fit_best.predict(X_test)

# Calculate performance
println("The model score (mean accuracy) on test set is: ", clf_model_fit_best.score(X_test, y_test))
mse = mean_squared_error(y_test, y_pred)
println("The mean squared error (MSE) on test set: ", mse)

# plot training curve
test_score = zeros((best_params["n_estimators"]))
for (i, y_pred) in enumerate(clf_model_fit_best.staged_predict(X_test))
    test_score[i] = accuracy_score(y_test, y_pred)
end

df_learningcurve1 = DataFrame([collect(range(1, best_params["n_estimators"])), clf_model_fit_best.train_score_, string.(ones(best_params["n_estimators"]))], ["n_estimators", "score", "train"])
df_learningcurve2 = DataFrame([collect(range(1, best_params["n_estimators"])), test_score, string.(zeros(best_params["n_estimators"]))], ["n_estimators", "score", "train"])
df_learningcurve = vcat(df_learningcurve1, df_learningcurve2)
plt_learningcurve = @vlplot(data=df_learningcurve)+
@vlplot(:line, x=:n_estimators, y=:score, color=:train)
plt_learningcurve

# # different approach for training curve
# train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_model, X_train, y_train, cv=10, return_times=true) # replace cv=30 with cv-pyobject
# @vlplot(:point, x=train_sizes, y=vec(mean(train_scores, dims=2))) # maybe dims=2, think abou it! https://thedatascientist.com/learning-curves-scikit-learn/

