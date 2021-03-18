import numpy as np
import pandas as pd
import Utils


class FeatureCreator:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        
    def get_model_data(self):
        '''
        If available, read model_data.parquet.gzip from cache
        Otherwise create model_data and write it to cache
        '''
        model_data = Utils.parquet_caching(
            parquet_name = "model_data",
            callback = self.create_features
        )
        self.model_data = model_data
        return model_data
        
        
    def create_features(self):
        dataset = self.dataset
        dataset = self.create_product_cat(dataset)
        dataset = self.create_substitue_discount(dataset)
        dataset = self.create_cat_cluster(dataset)
        dataset = self.dummy_encode_clusters(dataset)
        dataset = self.create_time_features(dataset)
        dataset = self.get_discount_features(dataset)
        dataset = self.get_elasticities(dataset)
        return dataset
    
    
    def create_product_cat(self, dataset):
        '''
        Add product category (e.g. drinks)
        EDA based feature, we know categories go in groups of 10
        '''
        eda_based_category_size = 10
        n_products = self.config['model']['n_products']
        product_range = n_products + 1
        label_range = int(n_products / eda_based_category_size)
        
        cut_bins = list(range(0, product_range, eda_based_category_size))
        cut_labels = list(range(0, label_range))
        dataset['product_cat'] = pd.cut(
            dataset['product'], 
            bins=cut_bins, 
            labels=cut_labels, 
            right=False
        )
        return dataset
  

    def create_substitue_discount(self, dataset):
        '''
        substitute discounts (e.g. Pepsi vs. Coca-Cola)
        calculate the sum of discounts a shopper was given for 
        substitute products (same category) in the same week
        '''
        partition = ['week', 'shopper', 'product_cat']
        dataset["cat_discount_sum"] = dataset.groupby(partition)['discount'].transform('sum')
        
        calc_substitue_discount = lambda row: row['cat_discount_sum'] - row['discount']

        dataset['substitue_discount'] = dataset.apply(calc_substitue_discount, axis=1)
        dataset = dataset.drop(columns=["cat_discount_sum"])
        return dataset


    def create_cat_cluster(self, dataset):
        '''
        add category cluster (e.g. junk food)
        EDA based feature, we found 3 category clusters (labels 0, 1, 2)
        '''
        cat_cluster_labels = [1, 1, 2, 2, 0, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2]
        cat_to_cluster_mapping = pd.DataFrame({"cat_cluster": cat_cluster_labels})
        dataset = dataset.merge(cat_to_cluster_mapping, left_on='product_cat', right_index=True)
        return dataset

    
    def dummy_encode_clusters(self, dataset):
        '''
        Dummy encode the category clusters (labels 0, 1, 2)
        '''
        one_hot = pd.get_dummies(dataset['cat_cluster'])
        one_hot.columns = ['cluster_0', 'cluster_1', 'cluster_2']
        dataset = dataset.join(one_hot)
        dataset = dataset.drop(columns=['cat_cluster'])
        return dataset

    
    def create_time_features(self, dataset):
        time_factor = 1.15
        max_weeks = np.ceil(dataset["week"].max() * time_factor)
        
        def get_weeks_since_last_purchase(row):
            '''
            Calculate n weeks since last product purchase by same shopper
            '''
            current_week = row['week']
            week_hist = row['week_hist']
            if not isinstance(week_hist, list): return max_weeks  
            past_purchase_weeks = [i for i in week_hist if i < current_week]
            if not past_purchase_weeks: return max_weeks
            last_pruchase_week = past_purchase_weeks[-1]
            weeks_since_last_purchase = current_week - last_pruchase_week
            return weeks_since_last_purchase
        
        dataset['weeks_since_last_purchase'] = dataset.apply(get_weeks_since_last_purchase, axis=1)
    
        def get_trend(row, window):
            '''
            Calculate relative product purchase frequency for varying time windows
            '''
            week = row['week']
            week_hist = row['week_hist']
            if not isinstance(week_hist, list): return 0  
            purchases_in_window = [i for i in week_hist if week - window <= i < week]
            trend = len(purchases_in_window) / window
            return trend

        for window in self.config['model']['trend_windows']:
            dataset["trend_" + str(window)] = dataset.apply(get_trend, args=([window]), axis=1)
        
        dataset["product_freq"] = dataset.apply(lambda row: get_trend(row, row['week']), axis=1)
        return dataset
    
    
    def get_discount_features(self, dataset):
        '''
        Calculate discount features
        '''
        def create_shopper_product_redemption_rate(row):
            discount_redeemed = row['discount_redeemed_weeks'] 
            discount_received = row['discount_received_weeks'] 
            week = row['week']
    
            try:
                redeemed = [redeem_week for redeem_week in discount_redeemed if redeem_week < week]
                received = [receive_week for receive_week in discount_received if receive_week < week]
                n_redeemed = len(redeemed)
                n_received = len(received)
                shopper_redemption_rate = n_redeemed / n_received
            except TypeError: # shopper never received or redeemed a coupon
                shopper_redemption_rate = 0
            except ZeroDivisionError: # shopper never received or redeemed a coupon prior to the current week
                shopper_redemption_rate = 0
            return shopper_redemption_rate
        
        def create_discount_buy(row): 
            discount_received = row['discount_received_weeks']
            product_purchased = row['week_hist']
            week = row['week']
    
            try:
                received = [receive_week for receive_week in discount_received if receive_week < week]
                purchased = [purchase_week for purchase_week in product_purchased if purchase_week < week]
                n_received = len(received)
                n_purchased = len(purchased)
                discount_buy = n_received / n_purchased
            except TypeError: # shopper never received a coupon or purchased the product
                discount_buy = 0
            except ZeroDivisionError: # shopper never received a coupon or purchased the product prior to the current week
                discount_buy = 0
            return discount_buy
        
        data = dataset.copy()
        data['discount_redeemed_weeks'] = data['discount_redeemed_weeks'].apply(lambda l: l if isinstance(l, list) else [])
        data['discount_received_weeks'] = data['discount_received_weeks'].apply(lambda l: l if isinstance(l, list) else [])
        
        shopper_redemption = data.groupby('shopper')[['discount_received_weeks', 'discount_redeemed_weeks']].sum().applymap(len)
        shopper_redemption = shopper_redemption['discount_redeemed_weeks'] / shopper_redemption['discount_received_weeks']
        shopper_redemption.name = 'shopper_redemption_rate'
        
        product_redemption = data.groupby('product')[['discount_received_weeks', 'discount_redeemed_weeks']].sum().applymap(len)
        product_redemption = product_redemption['discount_redeemed_weeks'] / product_redemption['discount_received_weeks']
        product_redemption.name = 'product_redemption_rate'
        
        dataset['shopper_product_redemption_rate'] = dataset.apply(create_shopper_product_redemption_rate, axis=1)
        dataset['discount_buy'] = dataset.apply(create_discount_buy, axis=1)
        dataset = dataset.merge(shopper_redemption, left_on='shopper', right_index=True)
        dataset = dataset.merge(product_redemption, left_on='product', right_index=True)
        
        dataset = dataset.drop(columns=['week_hist', 'discount_received_weeks', 'discount_redeemed_weeks'])
        dataset = dataset.sort_index()
        return dataset

    
    def get_elasticities(self, dataset):
        test_week = self.config['model']['test_week']
        n_products = self.config['model']['n_products']

        product_elasticities = pd.DataFrame({ 'product' : range(n_products)})
        elast = []

        train = dataset[dataset['week'] < test_week].copy()
        for i in range(n_products):
            product = train[train['product'] == i]

            reg_price_all= product[product['discount'] == 0]
            discount_30_all = product[product['discount'] == 0.3]

            reg_price_offer = len(reg_price_all)
            discount_30_offer = len(discount_30_all)

            reg_price_buy = len(reg_price_all[reg_price_all['purchased'] == 1])
            discount_30_buy = len(discount_30_all[discount_30_all['purchased'] == 1])

            reg_price_buy_rate = reg_price_buy / reg_price_offer
            discount_buy_rate = discount_30_buy / discount_30_offer

            elast.append((discount_buy_rate - reg_price_buy_rate) / (0.3 * reg_price_buy_rate))


        product_elasticities['elasticities'] = elast
        dataset = dataset.merge(product_elasticities, how='left')

        return dataset