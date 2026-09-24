from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


def get_model_pipeline(preprocess):

    model = LinearRegression()

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    return pipeline