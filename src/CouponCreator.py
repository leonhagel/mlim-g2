import pandas as pd
import numpy as np

class CouponCreator():

    def __init__(self, model, config):
        
        self.model = model
        self.config = config
        self.revenues = None
        self.top_coupons = None

    def get_top_coupons(self):
        revenue = self.get_revenue()
        top_coupons = self.create_top_coupons(revenue)
        return top_coupons
        
    # expected revenue
    # ----------------------------------------------------------------------------------
    def get_revenue(self):
        
        discounts = self.config['model']['discounts']
        X_template = self.model.X_test.copy()
        X_template['discount'] = None
        X_template['substitue_discount'] = 0
        
        discount_values = [0] + discounts
        X = self._add_discounts(X_template, discount_values)
        # calculating the expected revenue
        revenue = self.model.data.loc[X.index, ['week', 'shopper', 'product', 'price', 'product_cat']]
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
        X = pd.DataFrame()
        template = X_template.copy()
        for discount in discounts:
            template["discount"] = discount
            X = X.append(template)
        return X

    
    # extracting top coupons
    # ----------------------------------------------------------------------------------
    def create_top_coupons(self, revenue):
        n_coupons = self.config['model']['n_coupons']
        shoppers = range(self.config['model']['n_shoppers'])
        # calculating the top coupons
        coupons = {}
        for shopper in shoppers:
            coupons[shopper] = self._create_top_coupons(revenue, shopper, n_coupons)    
        # creating the final output
        output = list(coupons.values())[0]
        output["coupon"] = range(n_coupons)
        for coupon in list(coupons.values())[1:]:
            coupon["coupon"] = range(n_coupons)
            output = output.append(coupon)
        output = output[["shopper", "week", "coupon", "product", "discount"]]
        # converting dtypes
        output['shopper'] = output['shopper'].astype('int64')
        output['week'] = output['week'].astype('int64')
        output['coupon'] = output['coupon'].astype('int64')
        output['product'] = output['product'].astype('int64')
        output['discount'] = output['discount'].astype('float64')
        
        self.top_coupons = output
        return output

    def _create_top_coupons(self, revenue, shopper, n_coupons):
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
            # do not add a coupon if it would be a second coupon for a product
            condition1 = np.logical_not(np.isin(row["product"], output["product"]))
            # do not add a coupon if it would be a second coupon for a product_cat (i.e. two substitutes receive discounts)
            condition2 = np.logical_not(np.isin(row["product_cat"], output["product_cat"]))
            if condition1 and condition2:
                output = output.append(row) 
            i += 1
        return output
    
