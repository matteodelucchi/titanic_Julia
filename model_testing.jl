using ScikitLearn
using DataFrames
using CSV
using Statistics
using Titanic

@sk_import tree: DecisionTreeClassifier
@sk_import ensemble: RandomForestClassifier
@sk_import model_selection: cross_val_score
@sk_import model_selection: GridSearchCV

# Load the data
df_train = DataFrame(CSV.File("./data/X_train_enc.csv"))
df_test = DataFrame(CSV.File("./data/X_test_enc.csv"))

# Split the data in X and y parameters
X = df_train[!, Not(:Survived)]
y = df_train[!, :Survived]


# Convert into names and into a feature matrixs
X_names = names(X)
X_features = Matrix(X)


# Define a random seed for reproduction
SEED = 15041912 # Date when the Titanic sank

# Create a tree model
tree = DecisionTreeClassifier(
    criterion = "gini", 
    min_samples_split = 4, 
    random_state = SEED, 
    min_samples_leaf = 5, 
    max_depth = 5, 
    max_features = "sqrt"
)

# Creat the random forest model
forest = RandomForestClassifier(
    n_estimators = 1000,
    criterion = "gini",
    max_depth = 5, 
    min_samples_split = 4,
    min_samples_leaf = 5,
    max_features = "sqrt", 
    random_state = SEED,
    oob_score = true,
    n_jobs=-1,
    verbose=1    
)



# Fit the models
fit!(tree, X_features, y)
fit!(forest, X_features, y)


# Check the important features
decisitontree = hcat(tree.feature_importances_, X_names)
println("Decision Tree Classifier - Best Features")
decisitontree[sortperm(decisitontree[:, 1], rev=true), :]


rand_forest = hcat(forest.feature_importances_, X_names)
println("Random Forest Classifier - Best Features")
rand_forest[sortperm(rand_forest[:,1], rev=true), :]




# Prepare the data for the y check
X_test = Matrix(df_test[!, Not(:Survived)])
y_test = df_test[!, :Survived]


# Predict the y values
y_tree = tree.predict(X_test)
y_forest = forest.predict(X_test)


# Checking for accuracy
accuracy_tree = count(y_test .== y_tree)/length(y_test)
accuracy_forest = count(y_test .== y_forest)/length(y_test)


## Do some cross-CrossValidation
# CrossValidation on Tree Model
mean(cross_val_score(tree, X_features, y, cv = 10))

# Make grid check on Random Forest
param_grid = Dict(
    "n_estimators" => [5, 10, 50, 100, 200, 500, 1000], 
    "max_depth" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split" => [2, 3, 4, 5, 6, 7],
    "min_samples_leaf" => [2, 3, 4, 5, 6, 7]
)

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5, error_score="raise")

# Fit it on the model
fit!(grid_forest, X_features, y)

# Make the predictions
y_gforest = predict(grid_forest, X_test)

# Get the accuarcy
accuracy_gridforest = count(y_test .== y_gforest)/length(y_test)



# Using the best features from the CV on the old Forest model
grid_forest.best_params_

forest2 = RandomForestClassifier(
    n_estimators = 5,
    criterion = "gini",
    max_depth = 4, 
    min_samples_split = 3,
    min_samples_leaf = 4,
    max_features = "sqrt", 
    random_state = SEED,
    oob_score = true,
    n_jobs=-1,
    verbose=1    
)

fit!(forest2, X_features, y)

rand_forest2 = hcat(forest2.feature_importances_, X_names)
println("Random Forest Classifier - Best Features after CV")
rand_forest2[sortperm(rand_forest2[:,1], rev=true), :]

y_forest2 = forest2.predict(X_test)


accuracy_forest2 = count(y_test .== y_forest2)/length(y_test)


# Apply the model on the acutal test data
#Load
df_titanic = DataFrame(CSV.File("./data/df_test_enc.csv"))

# Make matrix
df_titanic = Matrix(df_titanic)

# Do the predictions
y_titanic = forest.predict(df_titanic)



function predict_submission(model, X_test)
    passengerid = DataFrame(CSV.File("./data/test.csv"))[:,1]
    y_pred = model.predict(X_test)

    df_pred = DataFrame([passengerid, y_pred], ["PassengerId", "Survived"])

    CSV.write("./data/DOM_submission_tree.csv", df_pred)
end

# Create DataFrame and create CSV file
predict_submission(forest, df_titanic)