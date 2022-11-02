using ScikitLearn
using CSV
using DataFrames
using Statistics
using VegaLite
@sk_import model_selection: learning_curve
@sk_import inspection: permutation_importance
@sk_import metrics: mean_squared_error
@sk_import model_selection: train_test_split
@sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/train.csv")) # TODO: Select preprocessed file!
df_train = dropmissing(df_train[:, [2,3,6,7,8,10]])
# X = df_train[:, 3:end] # This must be in Matrix format
X = Matrix(df_train[:, 2:end])
y = df_train[:, 2]
# y = reshape(y, length(y), 1)

# Split train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

# set learning parameters
params = Dict(
    "n_estimators"=> 500,
    "max_depth"=> 4,
    "min_samples_split"=> 5,
    "learning_rate"=> 0.01,
    "loss"=> "squared_error",
)

# Fit a classifier
clf_model = GradientBoostingClassifier(n_estimators = params["n_estimators"],
    max_depth = params["max_depth"],
    min_samples_split = params["min_samples_split"],
    learning_rate = params["learning_rate"])
clf_model_fit = clf_model.fit(X_train, y_train)

# Calculate performance
mse = mean_squared_error(y_test, clf_model_fit.predict(X_test))
print("The mean squared error (MSE) on test set: ", mse)

# plot training curve
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_model, X_train, y_train, cv=30,return_times=true) # replace cv=30 with cv-pyobject

@vlplot(:point, x=train_sizes, y=vec(mean(train_scores, dims=2))) # maybe dims=2, think abou it! https://thedatascientist.com/learning-curves-scikit-learn/

