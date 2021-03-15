import pandas as pd
import category_encoders
import sklearn
import lightgbm # https://github.com/microsoft/LightGBM/issues/1369


class Model:
    
    def __init__(self, model_data):
        self.data = model_data
    

    def train_test_split(self, config):
        """
        split data into X_train, y_train, X_test, y_test
        The size of X_train is determined by the train window (weeks)
        We perform Weight of Evidence Encoding (WOE) for shoppers and products
        We separating target (purchased) and features
        """
        test_week = config['model']['test_week']
        train_window = config['model']['train_window']
        data = self.data
        

        # Split test_week and train_weeks
        # ------------------------------------------------------------------------------
        start = test_week - train_window
        train = data[(data["week"] >= start) & (data["week"] < test_week)]
        test = data[data["week"] == test_week]
        train, test = self.woe_encode(train, test, 'shopper')
        train, test = self.woe_encode(train, test, 'product')
        
        # Split features X and target y
        # ------------------------------------------------------------------------------
        non_features = ["shopper", "week", "product", "purchased"]
        features = [col for col in train.columns if col not in non_features]

        X_train = train[features]
        y_train = train["purchased"]
        X_test = test[features]
        y_test = test["purchased"]

        return X_train, y_train, X_test, y_test

    
    def woe_encode(self, train, test, feature):
        train = train.copy()
        test = test.copy()
        encoder = category_encoders.WOEEncoder()
        
        train[f'{feature}_WOE'] = encoder.fit_transform(
            train[feature].astype("category"), train["purchased"]
        )[feature].values
        
        test[f'{feature}_WOE'] = encoder.transform(
            test[feature].astype("category")
        )[feature].values
        return train, test
                

    def fit(self, X_train, y_train, **kwargs):
        """
        Fit a simple LGBM Classifier
        """
        lgbm_classifier = lightgbm.LGBMClassifier()
        lgbm_classifier.fit(X_train, y_train, **kwargs)
        self.lgbm_classifier = lgbm_classifier

    
    def predict(self, X_test):
        """
        Compute purchase probability predictions
        """
        lgbm_classifier = self.lgbm_classifier
        y_hat = lgbm_classifier.predict_proba(X_test)[:, 1]
        return y_hat

    
    def get_score(self, y, y_hat):
        """
        Calculate log_loss score
        """
        metric = sklearn.metrics.log_loss
        return metric(y, y_hat)