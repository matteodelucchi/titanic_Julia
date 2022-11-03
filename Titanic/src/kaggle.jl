function predict_submission(model, X_test)
    passengerid = DataFrame(CSV.File("./data/test.csv"))[:,1]
    y_pred = model.predict(X_test)

    df_pred = DataFrame([passengerid, y_pred], ["PassengerId", "Survived"])

    CSV.write("./data/submission.csv", df_pred)
end