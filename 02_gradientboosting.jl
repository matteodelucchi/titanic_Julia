using ScikitLearn
using CSV
using DataFrames
@sk_import inspection: permutation_importance
@sk_import metrics: mean_squared_error
@sk_import model_selection: train_test_split
@sk_import ensemble: GradientBoostingClassifier

# Read in preprocessed data
df_train = DataFrame(CSV.File("./data/train.csv")) # TODO: Select preprocessed file!
# X = df_train[:, 3:end] # This must be in Matrix format
X = Matrix(df_train[:, [3,6,7,8,10]])
y = df_train[:, 2]
y = reshape(y, length(y), 1)

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

# Make a classifier
clf_model = GradientBoostingClassifier(n_estimators = params["n_estimators"],
    max_depth = params["max_depth"],
    min_samples_split = params["min_samples_split"],
    learning_rate = params["learning_rate"],
    loss = params["loss"])
clf_model_fit = clf.fit(X_train, y_train)