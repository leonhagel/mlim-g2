import pandas as pd
import numpy as np


# class inheritance level: 3 - Determining the optimal coupons
# ==================================================================================
from Pipeline import Purchase_Probabilities

# first approach: coupon effects only on the own purchase probability, no effects of
# the coupon on other products' probabilities, lha
class No_Cross_Effects(Purchase_Probabilities):
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        3

    goal:
        - use provided purchase probability model to generate the final output

    approach:
        - take the trained model
        - predict the probabilities for each shopper-product-discount combination
        - (not taking into account: discounts in complements or substitutes)

    functionality:
        - creating an overview/dataframe of expected revenue changes for candidate
          discounts
        - getting the top-n coupons for each shopper
        - pipeline function which summarizes the process
    """

    def __init__(self):
        """
        attributes:
            - discounts: list
                - list of candidate discounts
                - specified using self.get_revenues()
            - n_coupons: int
                - number of coupons to be recommended for each shopper
                - specified using self.pipeline(), self.get_top_coupons()
            - top_coupons: pd.DataFrame
                - dataframe of top-n coupons for each shopper
                - specified using self.get_top_coupons()

            - (helper-attributes: data, mappings)
            - (purchase-probabilities: shoppers, test_week, train_week, df, features, model_type, model)

        public methods:
            - get_revenue: creates an overview/dataframe of changes in expected revenue
              due to discounts
            - get_top_coupons: extracts the top-n coupons for each shopper based on the
              revenue data
            - pipeline: summarizes the greedy approach to determine the optimal coupons

            - (prices: get_price_map, aggregate_price_map)
            - (product_histories: get_history_map, get_history, get_last_purchase,
              get_trend)
            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
            - (purchase_probabilities: prepare, train_test_split, fit, predict, score)
        """
        super().__init__()
        self.discounts = None
        self.n_coupons = None
        self.top_coupons = None

        
    # changes in expected revenue
    # ----------------------------------------------------------------------------------
    def get_revenue(self, X_zero, discounts, df="prepare"):
        """
        use:
            - creating an overview/dataframe which contains the effects of discounts on
              expected revenues

        input:
            - X_zero: pd.DataFrame
                - dataframe with all shopper-product-test_week combinations; discount
                  feature needs to be set to zero, i.e. X_test output from
                  self.train_test_split with its discount feature set to 0
            - discounts: list
                - list of candidate discount values
            - df: pd.DataFrame or str
                - dataframe which has been the basis for model training
                - for str: dataframe needs to be stored at self.data[str]
                - default='prepare': uses the prepared data stored at self.data['prepare']

        return: pd.DataFrame
            - overview/dataframe which contains the effects of discounts on expected
              revenues
        """
        if type(df) == str:
            df = self.data[df]
        # adding candidate discounts to the test data
        X = self._add_discounts(X_zero, discounts)
        revenue = df.loc[X.index, ["shopper", "week", "product", "price"]].copy()
        # calculating the expected revenue
        revenue["discount"] = X["discount"].copy()
        revenue["probabilities"] = self.predict(self.model, X)
        revenue["exp_revenue"] = (
            revenue["probabilities"] * revenue["price"] * (1 - revenue["discount"])
        )
        # delta revenue: difference in expected revenue
        revenue["d_revenue"] = None
        for discount in discounts:
            revenue.loc[revenue["discount"] == discount, "d_revenue"] = (
                revenue.loc[revenue["discount"] == discount, "exp_revenue"]
                - revenue.loc[revenue["discount"] == 0, "exp_revenue"]
            )
        # sorting the dataframe by delta revenue
        revenue.sort_values(by="d_revenue", ascending=False)
        self.data["revenue"] = revenue
        return revenue

    def _add_discounts(self, X_zero, discounts):
        """
        use:
            - takes X_zero (i.e. X_test output from self.train_test_split with its
              discount feature set to 0) and concatenates all
              shopper-product-discount combinations

        input:
            - X_zero: pd.DataFrame
                - X_test output from self.train_test_split with its discount feature
                  set to 0
            - discounts: list
                - list of candidate discounts to be added to the dataframe

        return: pd.DataFrame
            - dataframe containing all shopper-product-discount combinations for the
              test week; to be used to predict purchase probabilities
        """
        self.discounts = discounts
        X = X_zero.copy()
        template = X_zero.copy()
        for discount in discounts:
            template["discount"] = discount
            X = X.append(template)
        return X

    
    # extracting top coupons
    # ----------------------------------------------------------------------------------
    def get_top_coupons(self, shoppers="all", n_coupons=5, df="revenue"):
        """
        use:
            - extracting the top-n coupons for each shopper

        requirements:
            - df='revenue': revenue dataframe needs to be stored at self.data['revenue']

        input:
            - shopper: list or tuple
                - list of shoppers for which the top coupons should be extracted
                - for tuple: specifies the first and last shopper in a shopper range
                - default='all': use all shoppers which are included in the data
            - n_coupons: int
                - number of coupons to be recommended for each shopper
            - df: pd.DataFrame or str
                - dataframe which contains the effects of discounts on the expected
                  revenue
                - for str: dataframe needs to be stored at self.data[str]
                - default='revenue': Will use the revenue dataframe stored at
                  self.data['revenue']

        return: pd.DataFrame
            - final output which contains the top-n coupons for each shopper in the
              test week

        """
        self.n_coupons = n_coupons
        if type(shoppers) == tuple:
            shoppers = range(shoppers[0], shoppers[-1] + 1)
        if shoppers == "all":
            shoppers = self.shoppers
        coupons = {
            shopper: self._get_top_coupons(shopper, n_coupons, df=df)
            for shopper in shoppers
        }

        output = list(coupons.values())[0]
        output["coupon"] = range(n_coupons)
        for coupon in list(coupons.values())[1:]:
            coupon["coupon"] = range(n_coupons)
            output = output.append(coupon)

        output = output[["shopper", "week", "coupon", "product", "discount"]]
        self.top_coupons = output
        return output

    def _get_top_coupons(self, shopper, n_coupons, df="revenue"):
        """
        use:
            - extracting the top-n coupons for one shopper

        requirements:
            - revenue dataframe needs to be stored at self.data['revenue']

        input:
            - shopper: int
                - shopper for which the top coupons should be extracted
             - n_coupons: int
                - number of coupons to be recommended for each shopper
            - df: pd.DataFrame or str
                - dataframe which contains the effects of discounts on the expected revenue
                - for str: dataframe needs to be stored at self.data[str]
                - default='revenue': Will use the revenue dataframe stored at
                  self.data['revenue']

        return: pd.DataFrame
            - dataframe of the top-n coupons for the shopper; to be used for the final output
        """
        
        if type(df) == str:
            df = self.data[df]

        df = (
            df.loc[df["shopper"] == shopper, :]
            .sort_values(by="d_revenue", ascending=False)
            .copy()
        )
        df = df.loc[df["discount"] != 0, :]
        output = pd.DataFrame(df.iloc[0]).T
        i = 1
        while output.shape[0] < n_coupons:
            row = df.iloc[i]
            if np.logical_not(np.isin(row["product"], output["product"])):
                output = output.append(row)
            i += 1
        return output

    # pipeline function
    # ----------------------------------------------------------------------------------
    def pipeline(
        self,
        test_week,
        train_window,
        discounts,
        n_coupons=5,
        shoppers="all",
        model_type="lgbm",
    ):
        """
        use:
            - creating the final output which contains the top-n coupons for each
              shopper for the test week

        input:
            - test_week: int
                - week for which the top-n coupons should be extracted
            - train_window: int
                - weeks prior to the test week, which should be used for model training
            - discount: list
                - list of candidate discounts
            - n_coupons: int
                - number of coupons to be recommended for each shopper
                - default=5: according to the task
            - shoppers: list or tuple
                - list of shoppers for which the top coupons should be extracted
                - for tuple: specifies the first and last shopper in a shopper range
                - default='all': use all shoppers which are included in the data
            - model_type: str
                - type of the model to be used for predicting the purchase probabilities
                - default='lgbm': lightgbm classifier

        return: pd.DataFrame
            - final output which contains the top coupons for each shopper in the
              test week
        """
        # train test split
        X_train, y_train, X_zero, _ = self.train_test_split(test_week, train_window)
        X_zero["discount"] = 0
        # fitting the model
        model = self.fit(model_type, X_train, y_train)
        score = self.score(y_train, self.predict(model, X_train))
        print(f"[discount] train-log-loss: {score}")

        # creating model input for the candidate discounts and calculating the expected revenues
        # X = self._add_discounts(X_zero, discounts)
        revenue = self.get_revenue(X_zero, discounts)
        top_coupons = self.get_top_coupons(shoppers=shoppers, n_coupons=n_coupons)
        return top_coupons
