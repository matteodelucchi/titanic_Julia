using Titanic
using CSV
using DataFrames
using ScikitLearn
@sk_import linear_model: LogisticRegression
@sk_import model_selection: StratifiedKFold
@sk_import metrics: f1_score
@sk_import model_selection: GridSearchCV
@sk_import preprocessing: PolynomialFeatures

train = DataFrame(CSV.File("./data/X_train_enc.csv"))
test = DataFrame(CSV.File("./data/X_test_enc.csv"))

df_test_enc = DataFrame(CSV.File("./data/df_test_enc.csv"))
df_test = DataFrame(CSV.File("./data/test.csv"))

X_train, X_test = train[!, Not(1,2)], test[!, Not(1,2)]
y_train, y_test = train[!, 1], test[!, 1]

df_test_enc = df_test_enc[!, Not(1)]

function Logistic_Regression(X_train, y_train; nsplits=5, scoring="f1", n_jobs=1, stratify=nothing)

    # build the model and grid search object
    model = LogisticRegression()
    parameters = Dict("random_state" => 0:1:5, "class_weight" => ("dict", "balanced"), "penalty" => ("l1", "l2","elasticnet","none"))
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
  
    # train the model
    fit!(gridsearch, X_train, y_train)
  
    best_estimator = gridsearch.best_estimator_
  
    return best_estimator, gridsearch
end
  
###

# run classifier 
best_estimator, gridsearch = Logistic_Regression(Matrix(X_train), y_train)
best_estimator

# Make predictions:
y_pred = predict(best_estimator, Matrix(X_test))


# Evaluate the predictions
f1_score(y_test, y_pred)

function predict_submission(model, X_test)
    passengerid = DataFrame(CSV.File("./data/test.csv"))[:,1]
    y_pred = model.predict(X_test)

    df_pred = DataFrame([passengerid, y_pred], ["PassengerId", "Survived"])

    CSV.write("./data/submission_logistic_regression.csv", df_pred)
end

predict_submission(best_estimator, Matrix(df_test_enc))