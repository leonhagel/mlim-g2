import numpy as np

class Evaluation:
    
    def __init__(self, model, config, optimal_coupons):
        self.model = model
        self.config = config
        self.optimal_coupons = optimal_coupons
        self.random_coupons = None
        self.zero_coupons = None

        
    def evaluate(self):
        zero_coupons = self.get_zero_coupons()
        random_coupons = self.get_random_coupons()
        optimal_coupons = self.optimal_coupons

        zero_revenue = self.get_expected_total_revenue(zero_coupons)
        random_revenue = self.get_expected_total_revenue(random_coupons)
        optimal_revenue = self.get_expected_total_revenue(optimal_coupons)

        print(f"\nexpected total revenue for zero coupons:\t{zero_revenue}")
        print(f"expected total revenue for random coupons:\t{random_revenue}")
        print(f"expected total revenue for optimal coupons:\t{optimal_revenue}\n")
    

    def get_random_coupons(self):
        random_coupons = self.optimal_coupons[['shopper', 'week', 'coupon']].copy()
        shoppers = list(range(self.config['model']['n_shoppers'])) 
        products = list(range(self.config['model']['n_products']))
        discounts = self.config['model']['discounts']
        n_coupons = self.config['model']['n_coupons']

        # for each shopper: random choice of products and discounts
        for shopper in shoppers:
            random_products = np.random.choice(products, n_coupons, replace=False)
            random_discounts = np.random.choice(discounts, n_coupons)
            random_coupons.loc[random_coupons['shopper'] == shopper, 'product'] = random_products
            random_coupons.loc[random_coupons['shopper'] == shopper, 'discount'] = random_discounts
        return random_coupons

    
    def get_zero_coupons(self):
        zero_coupons = self.optimal_coupons[['shopper', 'week', 'coupon']].copy()
        shoppers = list(range(self.config['model']['n_shoppers'])) 
        products = list(range(self.config['model']['n_products']))
        n_coupons = self.config['model']['n_coupons']
        
        # create the same output structure using random products and zero discount
        for shopper in shoppers:
            random_products = np.random.choice(products, 5, replace=False)
            zero_discount = 0
            zero_coupons.loc[zero_coupons['shopper'] == shopper, 'product'] = random_products
            zero_coupons.loc[zero_coupons['shopper'] == shopper, 'discount'] = zero_discount
        return zero_coupons
    
    
    def get_expected_total_revenue(self, coupons):
        revenue = self.model.data[['week', 'shopper', 'product', 'price']].copy()
        revenue = revenue.loc[revenue['week'] == self.config['model']['test_week']]
        revenue['discount'] = 0
        revenue.loc[coupons.index, 'discount'] = coupons['discount']

        X_test = self.model.X_test.copy()
        X_test['discount'] = 0
        X_test.loc[coupons.index, 'discount'] = coupons['discount']

        revenue['probabilities'] = self.model.predict(X_test)
        revenue['expected_revenue'] = revenue["probabilities"] * revenue["price"] * (1 - revenue["discount"])
        total_revenue = revenue['expected_revenue'].sum()
        return total_revenue