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
        
    def create_features(self):
        
        dataset = self.dataset
        
    
        # weeks since last purchase
        # ------------------------------------------------------------------------------
        time_factor = 1.15
        max_weeks = np.ceil(dataset["week"].max() * time_factor)

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