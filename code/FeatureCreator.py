import pandas as pd
import numpy as np
import Utils


class FeatureCreator:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        
    def run(self):
        model_data = Utils.parquet_loader(
            parquet_name = "model_data",
            path = self.config['data']['path'],
            callback = self.create_features
        )
        return model_data
        

    # add product category (e.g. drinks)
    # ------------------------------------------------------------------------------
    def create_product_cat(self, dataset):
        # EDA based feature, we know categories go in groups of 10
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
  

    # substitute discounts (e.g. Pepsi vs. Coca-Cola)
    # calculate the sum of discounts a shopper was given for 
    # substitute products (same category) in the same week
    # ------------------------------------------------------------------------------
    def create_substitue_discount(self, dataset):
        partition = ['week', 'shopper', 'product_cat']
        dataset["cat_discount_sum"] = dataset.groupby(partition)['discount'].transform('sum')
        
        calc_substitue_discount = lambda row: row['cat_discount_sum'] - row['discount']

        dataset['substitue_discount'] = dataset.apply(calc_substitue_discount, axis=1)
        dataset = dataset.drop(columns=["cat_discount_sum"])
        return dataset


    # add category cluster (e.g. junk food)
    # -----------------------------------------------------------------------------
    def create_cat_cluster(self, dataset):
        # EDA based feature, we found 3 category clusters (labels 0, 1, 2)
        cat_cluster_labels = [1, 1, 2, 2, 0, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2]
        cat_to_cluster_mapping = pd.DataFrame({"cat_cluster": cat_cluster_labels})
        dataset = dataset.merge(cat_to_cluster_mapping, left_on='product_cat', right_index=True)
        return dataset

    
    # calculate the sum of discounts a shopper was given for complement products 
    # (e.g. Chips & Coca-Cola) in the same week
    # ------------------------------------------------------------------------------
    # Todo rename cluster discount
    def create_complement_discount(self, dataset):
        partition = ['week', 'shopper', 'cat_cluster']
        dataset["cluster_discount_sum"] = dataset.groupby(partition)["discount"].transform('sum')
        
        calc_complement_discount = lambda row: row['cluster_discount_sum'] - row['discount']

        dataset['complement_discount'] = dataset.apply(calc_complement_discount, axis=1)
        dataset = dataset.drop(columns=["cluster_discount_sum"])
        return dataset

    
    # Dummy encode the category clusters (labels 0, 1, 2)
    # ------------------------------------------------------------------------------
    def dummy_encode_clusters(self, dataset):
        one_hot = pd.get_dummies(dataset['cat_cluster'])
        one_hot.columns = ['cluster_0', 'cluster_1', 'cluster_2']
        dataset = dataset.join(one_hot)
        dataset = dataset.drop(columns=['cat_cluster'])
        return dataset

    
    def create_features(self):
        
        dataset = self.dataset
        dataset = self.create_product_cat(dataset)
        dataset = self.create_substitue_discount(dataset)
        dataset = self.create_cat_cluster(dataset)
        dataset = self.create_complement_discount(dataset)
        dataset = self.dummy_encode_clusters(dataset)
        
        return dataset
        
        # weeks since last purchase
        # ------------------------------------------------------------------------------
        time_factor = 1.15
        max_weeks = np.ceil(dataset["week"].max() * time_factor)


        def get_weeks_since_last_purchase(row):
            current_week = row['week']
            week_hist = row['week_hist']

            if not isinstance(week_hist, list): return max_weeks  
            past_purchase_weeks = [i for i in week_hist if i < current_week]
            if not past_purchase_weeks: return max_weeks
            last_pruchase_week = past_purchase_weeks[-1]
            weeks_since_last_purchase = current_week - last_pruchase_week
            
            return weeks_since_last_purchase
        
        dataset['weeks_since_last_purchase'] = dataset.apply(get_weeks_since_last_purchase, axis=1)
        
        print("added feature: weeks_since_last_purchase")
    
    
        # purchase trend features
        # ------------------------------------------------------------------------------
        def get_trend(row, window):
            week = row['week']
            week_hist = row['week_hist']
            if not isinstance(week_hist, list): return 0  
            purchases_in_window = [i for i in week_hist if week - window <= i < week]
            trend = len(purchases_in_window) / window
            return trend

        for window in self.config['model']['trend_windows']:
            dataset["trend_" + str(window)] = dataset.apply(get_trend, args=([window]), axis=1)

        print("added feature: purchase trend features")
        
        
        # product purchase frequency (set week as window)
        # ------------------------------------------------------------------------------
        dataset["product_freq"] = dataset.apply(lambda row: get_trend(row, row['week']), axis=1)
        
        print("added feature: product_freq")
                
        
        # drop week_hist and week_prices (helpers to derive features)
        # ------------------------------------------------------------------------------
        self.model_data = dataset.drop(columns=['week_hist', 'week_prices'])
        
        return self.model_data
    
    

    
    
    # @BENEDIKT
    # feature: user features, e.g. user-coupon redemption rate
    # ------------------------------------------------------------------------------


    # @SASCHA
    # feature: product cluster, e.g. discount in complement/substitute
    # ------------------------------------------------------------------------------