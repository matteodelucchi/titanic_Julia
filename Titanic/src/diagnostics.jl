function plt_trainingcurve(best_params, clf_model_fit_best, X_valid, y_valid)
    test_score = zeros((best_params["n_estimators"]))
    for (i, y_pred) in enumerate(clf_model_fit_best.staged_predict(X_valid))
        test_score[i] = accuracy_score(y_valid, y_pred)
    end

    df_learningcurve1 = DataFrame([collect(range(1, best_params["n_estimators"])), clf_model_fit_best.train_score_, string.(ones(best_params["n_estimators"]))], ["n_estimators", "score", "train"])
    df_learningcurve2 = DataFrame([collect(range(1, best_params["n_estimators"])), test_score, string.(zeros(best_params["n_estimators"]))], ["n_estimators", "score", "train"])
    df_learningcurve = vcat(df_learningcurve1, df_learningcurve2)
    
    plt_learningcurve = @vlplot(data=df_learningcurve)+
    @vlplot(:line, x=:n_estimators, y=:score, color={:train, labels=["valid", "train"]})
    return plt_learningcurve

    # # different approach for training curve
    # train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_model, X_train, y_train, cv=10, return_times=true) # replace cv=30 with cv-pyobject
    # @vlplot(:point, x=train_sizes, y=vec(mean(train_scores, dims=2))) # maybe dims=2, think abou it! https://thedatascientist.com/learning-curves-scikit-learn/
end

function plt_featureimportances(clf_model_fit_best; feats = names(df_train[:,Not("Survived")]))
    feat_imp = clf_model_fit_best.feature_importances_
    
    # feats = names(df_train[:,Not("Survived")])
    df_featimp = DataFrame([feat_imp, feats], ["importance", "feature"])
    # df_featimp = sort(df_featimp, [:importance, order(:feature, rev=false)], rev=true)
    df_featimp_sorted = sort(df_featimp, [:importance], rev=true)
    
    plt_featimp = @vlplot(data=df_featimp_sorted)+
    @vlplot(:bar, y=:feature, x=:importance)

    return plt_featimp
end