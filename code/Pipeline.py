# ==================================================================================
#  Imports
# ==================================================================================
import os
import time
import itertools                           
import numpy as np
import pandas as pd
import category_encoders
import sklearn
import lightgbm # https://github.com/microsoft/LightGBM/issues/1369
from tqdm import tqdm
from IPython.display import clear_output
import Utils
tqdm.pandas()

"""
    Current Inheritance Model (needs refactoring):    
    
       Helper <-- Purchase_Probabilities <-- No_Cross_Effects
               
"""

# ==================================================================================
#  Helper Class
# ==================================================================================
# todo: rename to Utils and break inheritance
class Helper:
    """
    Expose utility methods
    """

    def __init__(self):
        """
        Attributes:
            data: (dict) holds all dataframes
            mappings: (dict) todo: what kind of mappings?
        """
        self.data = {}
        self.mappings = {}

        
    # read parquet files from disk and optimize memory consumption
    # ----------------------------------------------------------------------------------
    def load(self, files: dict):
        dataframes = {}
        for key, paths in files.items():
            for path in paths:
                name_with_extension = os.path.basename(path)
                name = os.path.splitext(name_with_extension)[0]
                data = pd.read_parquet(path)
                dataframes[name] = Utils.reduce_mem_usage(name_with_extension, data)
        self.data = dataframes


    # merge baskets and coupons
    # ----------------------------------------------------------------------------------    
    # we need to refactor this reading functionality and include it in Utils!
    def get_merged(self, filename='merged.parquet.gzip'):
        
        def merge():
            return self.data["baskets"].merge(self.data["coupons"], how="outer")

        self.data["merged"] = Utils.parquet_loader(
            parquet_name = "merged", 
            callback = merge
        )
        
        return self.data["merged"]
    
    # clean
    # ----------------------------------------------------------------------------------   
    def clean(self):
        """
        shoppers: reduces to range 2000
        discount: set missing values to 0
        prices: calculate prices without discounts
        purchased: create binary target (1 if it was included in baskets data)
        """
        max_shoppers = 2000
        df = self.data["merged"].copy()
        
        df = df[df['shopper'] < max_shoppers]
        df["discount"].fillna(0, inplace=True)
        df["discount"] = df["discount"] / 100
        df["price"] = df["price"] / (1 - df["discount"])
        df["purchased"] = df["price"].notna().astype("int8")
        
        self.data["clean"] = df
        return df



    # get mapping
    # ----------------------------------------------------------------------------------  
    def get_week_hist(self):
        """
        group data by product and shopper to get associated list of purchase weeks
        """
        df = self.data["clean"]
        n_rows = df["shopper"].nunique()
        n_cols = df["product"].nunique()
        
        table = pd.DataFrame(itertools.product(list(range(n_rows)), list(range(n_cols))))
        table.columns = ['shopper', "product"]
        
        df = df[df["purchased"] == 1]
        hist = df.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='week_hist')
        week_hist = table.merge(hist, how='left')
        self.week_hist = week_hist
        
        return week_hist

        '''
            *** @SASCHA: ***************************************************************
            - Possible to extend the function to purchase histories of product clusters
            - input: mapping='cluster_histories'
            - df: product clusters need to be added somehow
            - config: needs to be verified
            ****************************************************************************

        if mapping == 'cluster_histories'
        cluster_history_config = {
            'df': df, 
            'row_name': 'shopper', 
            'column_name': 'CLUSTER_FEATURE_NAME', 
            'value_name': 'week', 
            'initial_array': [-np.inf]
        }
        '''
    
    # Redem. rates

    # Elasticties

    # Product clusters


# ==================================================================================
#  Purchase_Probabilities Class
# ==================================================================================
class Purchase_Probabilities(Helper):
    """
    goal:
        - forecasting purchase probabilities

    functionality:
        - data preparation, i.e. final data cleaning, feature creation, feature selection,
          train test split
        - model training and evaluation
    """

    
    def __init__(self):
        """
        attributes:
            - shoppers: list
                - list of shoppers which are used for the model
            - test_week: int
                - test data: week for which the model should be tested
                - specified using self.train_test_split()
            - train_window: int
                - train data: number of week prior to the test week on which the model
                  will be trained
                - specified using self.train_test_split()
            - df: pd.DataFrame
                - input data for self.train_test_split()
                - specified using self.train_test_split()
            - features: list
                - names of features used for training the model
                - specified using self.train_test_split()
            - model_type: str
                - name of the model type used to specify which model should be used
                - specified using self.fit()
            - model: object
                - object of the trained model
                - specified using self.fit()

            - (helper-attributes: data, mappings)

        public methods:
            - prepare: prepares the data for model training - final data cleaning,
              feature creation
            - train_test_split: performs a time series split on the data + final
              preparation steps, feature selection
            - fit: trains the specified model
            - predict: predicts the specified model
            - score: calculates a model performance score
            - (product_histories: get_purchase_history, get_history, get_last_purchase,
              get_trend)
            - (helper: load, dump, reduce_data_size, get_merged_clean, save_mappings)
        """
        super().__init__()
        self.shoppers = None
        self.test_week = None
        self.train_window = None
        self.df = None
        self.features = None
        self.model_type = None
        self.model = None


    # get mode prices
    # ----------------------------------------------------------------------------------
    def get_mode_prices(self):
        """
        returns mode price for every product-week combination 
        table columns: product, week, mode_price
        """
        df = self.data['clean'].copy()
        df = df[df["price"].notna()]

        get_mode = lambda x: pd.Series.mode(x)[0]
        
        mode_prices = df.groupby(['product', 'week']).agg(
            mode_price=('price',get_mode)
        ).reset_index()

        return mode_prices
    

    def prepare(
        self,
        shopper = range(2000),
        week = range(86,91),
        product = range(250),
        trend_windows: list = [1, 3, 5],
    ):
        df = self.data['clean']
        """
        preparing data for train test split
            - adjusting the data structure according to the final output
            - treating missing price values using replacement
            - merge week_hist
            - feature creation
        """


        # adjusting the data structure according to the final output
        # ------------------------------------------------------------------------------
        output = pd.DataFrame(itertools.product(shopper, week, product))
        output.columns = ['shopper', 'week', 'product']        
        output = output.merge(df, how="left")
        output["purchased"].fillna(0, inplace=True)
        output["discount"].fillna(0, inplace=True)
        
        
        # replace missing prices with mode price in associated week
        # ------------------------------------------------------------------------------        
        mode_prices = self.get_mode_prices()
        
        output = output.merge(mode_prices, how='left', on=['week', 'product'])
        output['price'] = output['price'].fillna(output['mode_price'])
        output.drop('mode_price', axis=1, inplace=True)
        
        print("replaced missing prices with mean")    
    
    
        # merge week_hist column to derive features, will be dropped later
        # -> week_hist columns are: shopper, product, week_hist
        # -> merge is performed on product, shopper
        # ------------------------------------------------------------------------------
        output = output.merge(self.week_hist, how="left")

    
        # feature: weeks since last purchase
        # ------------------------------------------------------------------------------
        time_factor = 1.15
        max_weeks = np.ceil(output["week"].max() * time_factor)

        # this function can be improved / split
        def get_weeks_since_last_purchase(row):
            current_week = row['week']
            week_hist = row['week_hist']

            if not isinstance(week_hist, list): return max_weeks  
            past_purchase_weeks = [i for i in week_hist if i < current_week]
            if not past_purchase_weeks: return max_weeks
            last_pruchase_week = past_purchase_weeks[-1]
            weeks_since_last_purchase = current_week - last_pruchase_week
            
            return weeks_since_last_purchase
        
        output['weeks_since_last_purchase'] = output.progress_apply(get_weeks_since_last_purchase, axis=1)
        
        print("added feature: weeks_since_last_purchase")
    
    
        # feature: purchase trend features
        # ------------------------------------------------------------------------------
        def get_trend(row, window):
            week = row['week']
            week_hist = row['week_hist']
            if not isinstance(week_hist, list): return 0  
            purchases_in_window = [i for i in week_hist if week - window <= i < week]
            trend = len(purchases_in_window) / window
            return trend

        for window in trend_windows:
            output["trend_" + str(window)] = output.progress_apply(get_trend, args=([window]), axis=1)

        print("added feature: purchase trend features")
        
        #return output
        
        
        # feature: product purchase frequency (set week as window)
        # ------------------------------------------------------------------------------
        output["product_freq"] = output.progress_apply(lambda row: get_trend(row, row['week']), axis=1)
        
        print("added feature: product_freq")
        
        
        # @BENEDIKT
        # feature: user features, e.g. user-coupon redemption rate
        # ------------------------------------------------------------------------------

        
        # @SASCHA
        # feature: product cluster, e.g. discount in complement/substitute
        # ------------------------------------------------------------------------------

        
        print("prepare done")
        
        self.data["prepare"] = output
        return output

    
    def train_test_split(
        self,
        test_week: int,
        train_window: int,
        df="prepare",
    ):
        """
        use:
            - further preparing the data to be used for model training and evaluation

        functionality:
            - splits the data into train and test data according to the specified test
              week and train window
            - Weight of Evidence Encoding (WOE) for shoppers and products
            - separating target and features

        input:
            - test_week: int, week for which the model should be tested (test data)
            - train_window: int, number of weeks prior to the test week (train data)
            - df: pd.DataFrame or str
                - input dataframe on which the split should be performed
                - default='prepare': used the data prepared by self.prepare() stored at
                  self.data['prepare']

        return: pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
            - X_train, y_train, X_test, y_test to be used for model training
        """

        self.test_week = test_week
        self.train_window = train_window
        data = self.data[df] if (type(df) == str) else df

        
        # Split test_week and train_weeks
        # ------------------------------------------------------------------------------
        start = test_week - train_window
        train = data[(data["week"] >= start) & (data["week"] < test_week)]
        test = data[data["week"] == test_week]

        
        # WOE category encoding
        # ------------------------------------------------------------------------------
        encoder = category_encoders.WOEEncoder()
        
        train["shopper_WOE"] = encoder.fit_transform(
            train["shopper"].astype("category"), train["purchased"]
        )["shopper"].values
        
        test["shopper_WOE"] = encoder.transform(
            test["shopper"].astype("category")
        )["shopper"].values
        
        encoder = ce.WOEEncoder()
        
        train["product_WOE"] = encoder.fit_transform(
            train["product"].astype("category"), train["purchased"]
        )["product"].values
        
        test["product_WOE"] = encoder.transform(
            test["product"].astype("category")
        )["product"].values
        
        clear_output()

        
        # Split features X and target y
        # ------------------------------------------------------------------------------
        features_to_drop = [
            "purchased",
            "shopper",
            "week",
            "product",
            "product_history",
            "last_purchase",
        ]
        features = [col for col in train.columns if col not in features_to_drop]
        
        X_train = train[features]
        y_train = train["purchased"]
        X_test = test[features]
        y_test = test["purchased"]

        return X_train, y_train, X_test, y_test

    
    def fit(self, model_type: str, X_train, y_train, **kwargs):
        """
        use:
            - trains a model according to the specified model_type and train data

        input:
            - model_type: str
                - specifies which model should be used for model training
                - requirements: method will call a method called self._fit_#MODEL_TYPE#()
                  which needs to contain
                  all model training steps for the respective model_type and needs to
                  return the model object
            - X_train: pd.DataFrame
                - dataframe containing the train features
            - y_train: pd.DataFrame
                - dataframe containing the train target
            - **kwargs: keyword-argument dict
                - contains potential keyword arguments for the model training depending
                  on the model type

        return: object
            - trained model
        """
        self.model_type = model_type
        self.model = eval(f"self._fit_{model_type}(X_train, y_train, **kwargs)")
        return self.model

    
    def _fit_lgbm(self, X_train, y_train, **kwargs):
        """
        use:
            - trains a lightgbm classifier

        input:
            - X_train: pd.DataFrame
                - dataframe containing the train features
            - y_train: pd.DataFrame
                - dataframe containing the train target
            - **kwargs: keyword-argument dict
                - further meta parameter for the lgbm

        return: object
            - trained lgb-model object
        """

        model = lightgbm.LGBMClassifier()
        model.fit(X_train, y_train, **kwargs)
        return model

    
    def predict(self, model, X):
        """
        use:
            - predict a trained model based on the provided data

        input:
            - model: object
                - trained model
                - supported models: lightgbm classifier
            - X: pd.DataFrame
                - data for which the predictions should be made

        return: array-like
            - predicted purchase probabilities for the provided data
        """
        if type(model) == lightgbm.sklearn.LGBMClassifier:
            y_hat = model.predict_proba(X)[:, 1]
        return y_hat

    
    def score(self, y_true, y_hat, metric="log_loss"):
        """
        use:
            - calculating performance metrics for model evaluation

        input:
            - y_true: array-like
                - true value of the target
            - y_hat: array-like
                - predicted value of the target
            - metric: function(y_true, y_hat)
                - function that will calculate the metric/score
                - default='log-loss': calculates the log loss using sklearn.metrics.log_loss

        return:
            - model performance score
        """
        if metric == "log_loss":
            metric = sklearn.metrics.log_loss

        return metric(y_true, y_hat)
