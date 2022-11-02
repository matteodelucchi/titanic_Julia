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
@sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/train_cleaned.csv"))
df_train = df_train[:, Not("title")]
X = Matrix(df_train[:, 2:end])
y = df_train[:, 1]

# Split train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

# set learning parameters
params = Dict(
    "n_estimators"=> 500,
    "max_depth"=> 4,
    "min_samples_split"=> 5,
    "learning_rate"=> 0.01
)

# instantiate a CV scheme...
# TODO

# Fit a classifier
clf_model = GradientBoostingClassifier(n_estimators = params["n_estimators"],
    max_depth = params["max_depth"],
    min_samples_split = params["min_samples_split"],
    learning_rate = params["learning_rate"])
clf_model_fit = clf_model.fit(X_train, y_train)

# Calculate performance
println("The model score (mean accuracy) on test set is: ", clf_model_fit.score(X_test, y_test))
mse = mean_squared_error(y_test, clf_model_fit.predict(X_test))
println("The mean squared error (MSE) on test set: ", mse)

# plot training curve
test_score = zeros((params["n_estimators"]))
for (i, y_pred) in enumerate(clf_model_fit.staged_predict(X_test))
    test_score[i] = accuracy_score(y_test, y_pred)
end
df_learningcurve = DataFrame([collect(range(1, params["n_estimators"])), clf_model_fit.train_score_, test_score], ["n_estimators", "train_score", "test_score"])
plt_learningcurve = @vlplot(data=df_learningcurve)+
@vlplot(:line, x=:n_estimators, y={:train_score, label="training"}, color=:red)+
@vlplot(:line, x=:n_estimators, y={:test_score, label="testing"}, color=:blue)

df_learningcurve1 = DataFrame([collect(range(1, params["n_estimators"])), clf_model_fit.train_score_, string.(ones(params["n_estimators"]))], ["n_estimators", "score", "train"])
df_learningcurve2 = DataFrame([collect(range(1, params["n_estimators"])), test_score, string.(zeros(params["n_estimators"]))], ["n_estimators", "score", "train"])
df_learningcurve = vcat(df_learningcurve1, df_learningcurve2)
plt_learningcurve = @vlplot(data=df_learningcurve)+
@vlplot(:line, x=:n_estimators, y=:score, color=:train)
plt_learningcurve

# # different approach for training curve
# train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_model, X_train, y_train, cv=10,return_times=true) # replace cv=30 with cv-pyobject
# @vlplot(:point, x=train_sizes, y=vec(mean(train_scores, dims=2))) # maybe dims=2, think abou it! https://thedatascientist.com/learning-curves-scikit-learn/

