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
    
       Helper <-- Product_Histories <-- Purchase_Probabilities <-- No_Cross_Effects
               
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


    # mappings: initialize and create mappings
    # ----------------------------------------------------------------------------------
    def save_mappings(self, config: dict):
        """
        use:
            - creates mappings based on the provided configuration by looping over the
              data and storing the current value to the correct location in the map
            - stores the map to the 'mappings'-attribute

        input:
            - config: dict
                - key: name of the mapping to be created
                - values: kwargs:dict
                     - 'df': dataframe for which the map should be created
                     - 'row_name': name of the column in 'df' which will be the index
                       in the resulting dataframe/map
                     - 'column_name': name of the column in 'df' which will be the
                       column in the resulting dataframe/map
                     - 'value_name': name of the column in 'df' which is the feature of
                       interest. While iterating over the data, the value will be
                       appended to a list stored in the dataframe/map, located at
                       [row_name, column_name]
                     - 'initial_array': first elements in the value list
                         - default=[]: no first values

        return: pd.DataFrame
            - index: unique values of the 'row_name' column in 'df'
            - column: unique values of the 'column_name' column in 'df'
            - values: list of values for each index-column combination, based on the
              'value_name' column in 'df'
        """
        for name, cnfg in config.items():
            self.mappings[name] = self._get_mapping(**cnfg)


    # get mapping
    # ----------------------------------------------------------------------------------  
    def _get_mapping(
        self,
        df,
        row_name,
        column_name,
        value_name,
        initial_array: list = None,
    ):
        """
        create pivot table with array-like values
        example: maps a specific product and customer to the matching weeks of purchase
        products (rows) x shoppers (cols)  -> weeks (values saved in a list)
        Args:
            df: (pd.DataFrame) input data that shall be converted to a pivot table
            row_name: (str) df[row_name] will be the rows in the pivot table
            column_name: (str) df[column_name] will be the columns in the pivot table
            value_name: (str) df[value_name] will be list-aggregated values in the pivot table
            initial_array: (list) specify default values in an array 
        """
        if initial_array == None:
            initial_array = []

        n_rows = df[row_name].nunique()
        n_cols = df[column_name].nunique()
        
        table = pd.DataFrame(itertools.product(list(range(n_rows)), list(range(n_cols))))
        table.columns = [row_name, column_name]
        
        def aggregate(x):
            return initial_array + list(x) if (initial_array) else list(x)

        history = df.groupby([row_name, column_name]).agg({value_name:aggregate}).reset_index()
        merged = table.merge(history, how='left')
        merged[value_name] = merged[value_name].apply(lambda d: d if isinstance(d, list) else initial_array)
        pivot = merged.pivot_table(index=row_name, columns=column_name, values=value_name, aggfunc='first')
        
        return pivot
    
    
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
        mode_prices = df.groupby(['product', 'week']).agg(mode_price=('price',get_mode)).reset_index()

        return mode_prices


# ==================================================================================
#  Product_Histories Class
# ==================================================================================
class Product_Histories(Helper):
    """
    goal:
        - creating purchase history features, e.g. for each shopper-product combination

    functionality:
        - creating a purchase history maps
        - aggregating the purchase history to extract further history-based features,
          e.g. purchase trends, time since last purchase, etc.
    """

    def __init__(self):
        """
        attributes:
            - (helper-attributes: data, mappings)

        public methods:
            - get_purchase_history: creates a purchase history map
            - get_history: returns the purchase history for a given point in time
            - get_last_purchase: returns the last purchase for a given point in time
            - get_trend: returns the purchase trend for a given trend window and
              point in time

            - (helper: load, dump, reduce_data_size, get_merged_clean, save_mappings)
        """
        super().__init__()

    # creating purchase history maps
    # ----------------------------------------------------------------------------------
    def get_purchase_history(self, mapping: str = "product_histories"):
        """
        use:
            - creates the 'purchase_histories' map to be able to create history-based
              features
            - stores the purchase history map located at
              self.mappings['product_histories']

            *** @SASCHA: ***************************************************************
            - Possible to extend the function to purchase histories of product clusters
            - input: mapping='cluster_histories'
            - df: product clusters need to be added somehow
            - config: needs to be verified
            ****************************************************************************

        requirements:
            - data: requires either 'clean' or 'merged' data stored at self.data['clean']
              or self.data['merged'] respectively

        input:
            - mapping: str
                - name of the map to be created

        return: pd.DataFrame
            - 'product_histories' map to create further history-based features
              (index: shopper, columns: products, values: list of weeks in which each
              shopper purchased the respective product
        """

        # specifying the input data

        # df should be in args
        df = self.data["clean"]
        
        # reduce to purchased items only
        df = df[df["purchased"] == 1]
        
        purchase_history_config = {
            "df": df,
            "row_name": "shopper",
            "column_name": "product",
            "value_name": "week",
            "initial_array": [-np.inf],

        } 

        '''
        if mapping == 'cluster_histories'
        cluster_history_config = {
            'df': df, 
            'row_name': 'shopper', 
            'column_name': 'CLUSTER_FEATURE_NAME', 
            'value_name': 'week', 
            'initial_array': [-np.inf]
        }
        '''

        self.save_mappings({'product_histories': purchase_history_config})
        return self.mappings[mapping]

    # aggregating the history map for feature creation
    # ----------------------------------------------------------------------------------
    def get_history(
        self, 
        shopper: int, 
        product: int, 
        week: int, 
        mapping: str = "product_histories"
    ):
        """
        use:
            - extracting the relevant purchase history given a shopper, product and week

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        Args:
            shopper: (int) shopper id
            product: (int) product id
            week: (int) point in time for which the history should be returned
            mapping: (str) name of the relevant map stored at self.mappings[mapping]

        return: np.array
            - 'product_histories': array which contains all week numbers of purchase
              weeks for the provided shopper-product combination (prior to the provided week)
        """
        
        arr = np.array(self.mappings[mapping].loc[shopper, str(product)])
        return arr[arr < week]

    def get_last_purchase(
        self, 
        shopper: int, 
        product: int, 
        week: int, 
        mapping: str = "product_histories"
    ):
        """
        use:
            - receiving the last purchases week for the provided shopper-product
              combination at the provided point in time

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        input:
            - shopper: int
                - shopper id
            - product: int
                - product id
            - week: int
                - point in time for which the last purchase should be returned
            - mapping: str
                - name of the relevant map stored at self.mappings[mapping]
                - default='product_histories': returns shopper-product last purchases

        return: int
            - week of last purchase for the provided shopper-product combination and
              prior to the provided week
        """
        return self.get_history(shopper, product, week, mapping=mapping)[-1]

    def get_trend(
        self,
        shopper: int,
        product: int,
        week: int,
        trend_window: int,
        mapping: str = "product_histories",
    ):
        """
        use:
            - receiving a current purchase trend, i.e. purchase frequency over the
              specified trend window, for the provided shopper-product combination
              at the provided point in time

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        input:
            - shopper: int
                - shopper id
            - product: int
                - product id
            - week: int
                - point in time for which the last purchase should be returned
                        - mapping: str
            - trend_window: int
                - number of weeks prior to 'week' on which the trend calculations
            - mapping: str
                - name of the relevant map stored at self.mappings[mapping]
                - default='product_histories': returns shopper-product last purchases

        return: int
            - trend/purchase frequency for the provided trend window and
              shopper-product combination, prior to the provided week
        """

        history = self.get_history(shopper, product, week, mapping=mapping)
        return (
            np.unique(history[history >= week - trend_window]).shape[0] / trend_window
        )


# Redem. rates

# Elasticties

# Product clusters


# ==================================================================================
#  Purchase_Probabilities
# ==================================================================================

# class inheritance level: 2 - Predicting purchase probabilities
# ==================================================================================
class Purchase_Probabilities(Product_Histories):
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

        
    def prepare(
        self,
        #df = "clean", how do we handle df arguments?
        shopper = range(2000),
        week = range(86,91),
        product = range(250),
        trend_windows: list = [1, 3, 5],
    ):
        """
        use:
            - preparing the data for the train test split

        preparation operations:
            - adjusting the data structure according to the final output
            - treating missing price values using replacement
            - feature creation:
                - weeks since last purchase
                - purchase trends - based on the trend window
                - product purchase frequency - based on all available data

        requirements:
            - 'product_histories' map needs to be stored at
              self.mappings['product_histories']

        input:
            - df: pd.Dataframe or str
                - input dataframe on which the preparation operations should be performed
                - for str: input dataframe need to be stored at self.data[str]
                - default='clean': use the cleaned data frame stored at self.data['clean']
            - shopper: tuple or list
                - shopper ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a shopper id range
            - week: tuple or list
                - week ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a week id range
            - product: tuple or list
                - product ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a product id range
            - trend_windows: list
                - list of trend window weeks for which a trend feature should be created

        return: pd.DataFrame
            - prepared data to be used for the train test split
        """

        start = time.time()
        df = self.data['clean']

        
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
        
        print('ready!')
        return output
    

        # feature creation
        # ------------------------------------------------------------------------------
        print("prepare feature creation...")

        # weeks since last purchase
        output["weeks_since_last_purchase"] = output.progress_apply(
            lambda row: self.get_last_purchase(
                int(row["shopper"]), str(int(row["product"])), row["week"]
            ),
            axis=1,
        )
        output["weeks_since_last_purchase"] = (
            output["week"] - output["weeks_since_last_purchase"]
        )
        output.loc[
            output["weeks_since_last_purchase"] == np.inf, "weeks_since_last_purchase"
        ] = np.ceil(output["week"].max() * 1.15)

        # purchase trend features
        for window in trend_windows:
            output["trend_" + str(window)] = output.progress_apply(
                lambda row: self.get_trend(
                    int(row["shopper"]), str(int(row["product"])), row["week"], window
                ),
                axis=1,
            )
        # product purchase frequency
        output["product_freq"] = output.progress_apply(
            lambda row: self.get_trend(
                int(row["shopper"]), str(int(row["product"])), row["week"], row["week"]
            ),
            axis=1,
        )

        # **************************************************************
        # feature creation: user features, e.g. user-coupon redemption rate
        # benedikt
        # **************************************************************

        # **************************************************************
        # feature creation: product cluster features, e.g. discount in complement/substitute
        # sascha
        # **************************************************************

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
