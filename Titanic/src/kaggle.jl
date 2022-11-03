function predict_submission(model, X_test, filepath)
    passengerid = DataFrame(CSV.File("./data/test.csv"))[:,1]
    y_pred = model.predict(X_test)

    df_pred = DataFrame([passengerid, y_pred], ["PassengerId", "Survived"])

    CSV.write(filepath, df_pred)
end