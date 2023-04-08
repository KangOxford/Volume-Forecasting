import pandas as pd
def regularity_ols(X_train, y_train, X_test, regulator):
    if regulator == "OLS":
        # print("OLS")
        import statsmodels.api as sm
        def ols_with_summary(X, y):
            X = sm.add_constant(X, has_constant='add')
            results = sm.OLS(y, X).fit()
            # print(results.summary())
            return results

        def feature_selecting(model, X, y):
            selected_features = model.pvalues[1:].idxmax()
            while model.pvalues[selected_features] > 0.05:
                X = X.drop(selected_features, axis=1)
                if 'const' not in X.columns: X = sm.add_constant(X, has_constant='add')
                model = sm.OLS(y, X).fit()
                selected_features = model.pvalues[1:].idxmax()
            # print(model.summary())
            return model, X, y

        model = ols_with_summary(X_train, y_train)
        model, X, y = feature_selecting(model, X_train, y_train)
        columns = X.columns.to_list()
        # test sets
        if 'const' in columns:
            columns.remove('const')
            X = pd.DataFrame(X_test[columns]).T
            X = sm.add_constant(X, has_constant='add')
        y_pred = model.predict(X).values
        # assert type(y_pred) == np.float64
        return y_pred[0]
    elif regulator in ["Lasso", "Ridge"]:
        # print("LASSO / RIDGE")
        def find_best_regularity_alpha(X_train, y_train):
            if regulator == "Lasso":
                from sklearn.linear_model import LassoCV
                model = LassoCV(random_state=0, max_iter=10000000)
            if regulator == "Ridge":
                from sklearn.linear_model import RidgeCV
                model = RidgeCV(alphas=combined_array)
            model.fit(X_train, y_train)
            return model.alpha_

        best_regularity_alpha = find_best_regularity_alpha(X_train, y_train)
        # print(best_regularity_alpha) #$
        if regulator == "Lasso":
            from sklearn.linear_model import Lasso
            reg = Lasso(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        if regulator == "Ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        reg.fit(X_train, y_train)
        X = pd.DataFrame(X_test).T
        y_pred = reg.predict(X)
        return y_pred[0]
    else:
        raise NotImplementedError
