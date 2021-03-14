import pandas as pd
import category_encoders
import sklearn
import lightgbm # https://github.com/microsoft/LightGBM/issues/1369


class Model:
    
    def __init__(self, model_data, config):
        self.data = model_data
        self.config = config
    

    def train_test_split(self):
        """
        split data into X_train, y_train, X_test, y_test
        The size of X_train is determined by the train window (weeks)
        We perform Weight of Evidence Encoding (WOE) for shoppers and products
        We separating target (purchased) and features
        """
        test_week = self.config['model']['test_week']
        train_window = self.config['model']['train_window']
        data = self.data
        

        # Split test_week and train_weeks
        # ------------------------------------------------------------------------------
        start = test_week - train_window
        train = data[(data["week"] >= start) & (data["week"] < test_week)]
        test = data[data["week"] == test_week]
        train, test = woe_encoding(train, trest)
        
        # Split features X and target y
        # ------------------------------------------------------------------------------
        non_features = ["shopper", "week", "product", "purchased"]
        features = [col for col in train.columns if col not in non_features]

        X_train = train[features]
        y_train = train["purchased"]
        X_test = test[features]
        y_test = test["purchased"]

        return X_train, y_train, X_test, y_test

    
    def woe_encoding(self, train, test):
        encoder = category_encoders.WOEEncoder()
        
        train.loc[:,"shopper_WOE"] = encoder.fit_transform(
            train["shopper"].astype("category"), train["purchased"]
        )["shopper"].values
        
        test.loc[:,"shopper_WOE"] = encoder.transform(
            test["shopper"].astype("category")
        )["shopper"].values
        
        encoder = category_encoders.WOEEncoder()
        
        train.loc[:,"product_WOE"] = encoder.fit_transform(
            train["product"].astype("category"), train["purchased"]
        )["product"].values
        
        test.loc[:,"product_WOE"] = encoder.transform(
            test["product"].astype("category")
        )["product"].values
        
        encoder = ce.WOEEncoder()
        
        train.loc[:, "shopper_WOE"] = encoder.fit_transform(
            train["shopper"].astype("category"), train["purchased"]
        )["shopper"].values
        test.loc[:, "shopper_WOE"] = encoder.transform(
            test["shopper"].astype("category")
        )["shopper"].values
        encoder = ce.WOEEncoder()
        train.loc[:, "product_WOE"] = encoder.fit_transform(
            train["product"].astype("category"), train["purchased"]
        )["product"].values
        test.loc[:, "product_WOE"] = encoder.transform(
            test["product"].astype("category")
        )["product"].values
        
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