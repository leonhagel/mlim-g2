import pandas as pd
import numpy as np

class CouponCreator():

    def __init__(self, model):
        
        self.model = model
        self.revenues = None
        self.discounts = None
        self.n_coupons = None
        self.top_coupons = None

    def run(self, discounts, X_template, n_coupons=5):
        revenue = self.get_revenue(X_template, discounts)
        top_coupons = self.get_top_coupons(revenue, n_coupons=n_coupons)
        return top_coupons
        
    # expected revenue
    # ----------------------------------------------------------------------------------
    def get_revenue(self, X_template, discounts):
        
        discount_values = [0] + discounts
        X = self._add_discounts(X_template, discount_values)
        # calculating the expected revenue
        revenue = self.model.data.loc[X.index, ['week', 'shopper', 'product', 'price']]
        revenue['discount'] = X['discount']
        revenue["probabilities"] = self.model.predict(X)
        revenue["exp_revenue"] = revenue["probabilities"] * revenue["price"] * (1 - revenue["discount"])

        # delta revenue: difference in expected revenue
        revenue["delta_revenue"] = None
        for discount in discounts:
            revenue.loc[revenue["discount"] == discount, "delta_revenue"] = (
                revenue.loc[revenue["discount"] == discount, "exp_revenue"]
                - revenue.loc[revenue["discount"] == 0, "exp_revenue"]
            )
        
        # sorting the dataframe by delta revenue
        revenue.sort_values(by="delta_revenue", ascending=False)
        self.revenue = revenue
        return revenue

    def _add_discounts(self, X_template, discounts):
        
        self.discounts = discounts
        X = pd.DataFrame()
        template = X_template.copy()
        for discount in discounts:
            template["discount"] = discount
            X = X.append(template)
        return X

    
    # extracting top coupons
    # ----------------------------------------------------------------------------------
    def get_top_coupons(self, revenue, n_coupons=5):
        self.n_coupons = n_coupons
        # calculating the top coupons
        coupons = {}
        for shopper in range(250):
            coupons[shopper] = self._get_top_coupons(revenue, shopper, n_coupons)    
        # creating the final output
        output = list(coupons.values())[0]
        output["coupon"] = range(n_coupons)
        for coupon in list(coupons.values())[1:]:
            coupon["coupon"] = range(n_coupons)
            output = output.append(coupon)
        output = output[["shopper", "week", "coupon", "product", "discount"]]
        self.top_coupons = output
        return output

    def _get_top_coupons(self, revenue, shopper, n_coupons):
        # reduce data to the shopper in question
        revenue = (
            revenue.loc[revenue["shopper"] == shopper, :]
            .sort_values(by="delta_revenue", ascending=False)
            .copy()
        )
        revenue = revenue.loc[revenue["discount"] != 0, :]
        # extract the top-n coupons
        output = pd.DataFrame(revenue.iloc[0]).T
        i = 1
        while output.shape[0] < n_coupons:
            row = revenue.iloc[i]
            if np.logical_not(np.isin(row["product"], output["product"])):
                output = output.append(row)
            i += 1
        return output
    
