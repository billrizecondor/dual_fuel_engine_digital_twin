import numpy as np
    from sklearn.linear_model import LinearRegression

def train_exhaust_temp_model(df):
    """
    Trains a simple linear regression model to predict exhaust gas temperature
    based on power output.
    """
    df_clean = df[['power_output', 'exhaust_temp']].dropna()
    X = df_clean[['power_output']].values
    y = df_clean['exhaust_temp'].values

    model = LinearRegression()
    model.fit(X, y)

    return model