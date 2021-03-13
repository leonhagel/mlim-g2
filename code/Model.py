import pandas as pd
import category_encoders
from IPython.display import clear_output
import sklearn
import lightgbm # https://github.com/microsoft/LightGBM/issues/1369


class Model:
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
    

    def train_test_split(self):
        
        """
        split data into X_train, y_train, X_test, y_test
        The size of X_train is determined by the train window (weeks)
        We perform Weight of Evidence Encoding (WOE) for shoppers and products
        We separating target (purchased) and features
        Args:
            test_week: (int) week for which the model should be tested (test data)
            train_window: (int) number of weeks prior to the test week (train data)
        """

        test_week = self.config['model']['test_week']
        train_window = self.config['model']['train_window']
        data = self.data
        

        # Split test_week and train_weeks
        # ------------------------------------------------------------------------------
        start = test_week - train_window
        train = data[(data["week"] >= start) & (data["week"] < test_week)]
        test = data[data["week"] == test_week]


        # WOE category encoding
        # Hier wird ein Deprecation Warning geschmissen
        # FutureWarning: is_categorical is deprecated and will be removed in a future version.  
        # Use is_categorical_dtype instead
        # ------------------------------------------------------------------------------
        encoder = category_encoders.WOEEncoder()
        
        train["shopper_WOE"] = encoder.fit_transform(
            train["shopper"].astype("category"), train["purchased"]
        )["shopper"].values
        
        test["shopper_WOE"] = encoder.transform(
            test["shopper"].astype("category")
        )["shopper"].values
        
        encoder = category_encoders.WOEEncoder()
        
        train["product_WOE"] = encoder.fit_transform(
            train["product"].astype("category"), train["purchased"]
        )["product"].values
        
        test["product_WOE"] = encoder.transform(
            test["product"].astype("category")
        )["product"].values
        
        #clear_output()

        
        # Split features X and target y
        # ------------------------------------------------------------------------------
        non_features = ["shopper", "week", "product", "purchased"]
        features = [col for col in train.columns if col not in non_features]

        X_train = train[features]
        y_train = train["purchased"]
        X_test = test[features]
        y_test = test["purchased"]

        return X_train, y_train, X_test, y_test

    
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