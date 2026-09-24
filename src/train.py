import os
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(pipeline, X, y):
    
    mlflow.set_tracking_uri("file:./mlruns")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=kfold, scoring="r2"
    )

    print("CV Scores:", cv_scores)
    print("Average CV R2:", np.mean(cv_scores))


    param_grid = {
    "model__fit_intercept": [True, False],
    "model__positive": [True, False]
}  

    grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="r2"
)  
    with mlflow.start_run():

       
        grid_search.fit(X_train, y_train)

        best_model =  grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("num_rows", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])

     
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("CV_R2_mean", np.mean(cv_scores))
        

        print("\n------ MODEL METRICS ------")
        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

      
        mlflow.sklearn.log_model(
           sk_model=best_model,
           name="model",
           registered_model_name="house_price_model",
           skops_trusted_types=["numpy.dtype"]
)
    return best_model