from Pipeline import Helper, Purchase_Probabilities

# first apprach: coupon effects only on the own purchase probability, no effects of the coupon on other products' probabilities, lha 
class No_Cross_Effects(Purchase_Probabilities):
    
    def __init__(self):
        super().__init__()
        self.top_coupons = None
        self.discounts = None
        self.top_n_discounts= None, 
        self.shoppers=None
        
        
        
    def _add_discounts(self, X_zero, discounts):
        self.discounts = discounts
        X = X_zero.copy()
        template = X_zero.copy()
        for discount in discounts:
            template['discount'] = discount
            X = X.append(template)
        return X
    
    
    def get_revenue(self, X, df='prepare'):
        if type(df) == str:
            df = self.data[df]
            
        revenue = df.loc[X.index, ['shopper', 'week', 'product', 'price']].copy()
        revenue['discount'] = X['discount'].copy()
        revenue['probabilities'] = self.predict(self.model, X)
        revenue['exp_revenue'] = revenue['probabilities'] * revenue['price'] * (1 - revenue['discount'])
        # delta revenue: difference in expected revenue 
        discounts = revenue['discount'].unique()
        revenue['d_revenue'] = None
        for discount in discounts:
            revenue.loc[revenue['discount'] == discount, 'd_revenue'] = revenue.loc[revenue['discount'] == discount, 'exp_revenue'] - revenue.loc[revenue['discount'] == 0, 'exp_revenue']
        revenue.sort_values(by='d_revenue', ascending=False)
        self.data['revenue'] = revenue
        return revenue
    
    
    def _get_top_coupons(self, shopper, n_discounts, df='revenue'):
        import pandas as pd
        import numpy as np
    
        if type(df) == str:
            df = self.data[df]

        df = df.loc[df['shopper']==shopper, :].sort_values(by='d_revenue', ascending=False).copy()
        df = df.loc[df['discount'] != 0, :]
        output = pd.DataFrame(df.iloc[0]).T
        i=1
        while output.shape[0] < n_discounts:
            row = df.iloc[i]
            if np.logical_not(np.isin(row['product'], output['product'])):
                output = output.append(row)
            i += 1
        return output
            
    def get_top_coupons(self, shoppers=(0, 1999), n_discounts=5, df='revenue'):
        self.n_discounts= n_discounts
        shoppers = range(shoppers[0], shoppers[-1]+1) if type(shoppers) != list else shoppers
        self.shoppers = shoppers
        coupons = {shopper: self._get_top_coupons(shopper, n_discounts, df=df) for shopper in shoppers}

        output = list(coupons.values())[0]
        output['coupon'] = range(n_discounts)
        for coupon in list(coupons.values())[1:]:
            coupon['coupon'] = range(n_discounts)
            output = output.append(coupon)
    
        output = output[['shopper', 'week', 'coupon', 'product', 'discount']]
        self.top_coupons = output
        return output
    
    
    def pipeline(self, test_week, train_window, discounts, n_discounts=5, shoppers=(0, 1999), model_type='lgbm'):
        # train test split
        X_train, y_train, X_zero, _ = self.train_test_split(test_week, train_window)
        X_zero['discount'] = 0
        # fitting the model
        model = self.fit(model_type, X_train, y_train)
        score = self.score(y_train, self.predict(model, X_train))
        print(f"[discount] train-log-loss: {score}")
    
        # creating model input for the candidate discounts and calculating the expected revenues
        X = self._add_discounts(X_zero, discounts)
        revenue = self.get_revenue(X)
        top_coupons = self.get_top_coupons(shoppers=shoppers, n_discounts=n_discounts)
        return top_coupons