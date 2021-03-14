import pandas as pd
import numpy as np
import time
import os
for dirname, _, filenames in os.walk('/mlim-g2/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

class Elasticities:
    """
    load baskets and coupons df and merge every 
    week to calculate weekly price elasticities
    """

    def __init__(self):
        self.baskets = pd.read_parquet( r'../data/baskets.parquet', engine = 'pyarrow')
        self.coupons = pd.read_parquet( r'../data/coupons.parquet', engine = 'pyarrow' )
        
    def get_elasticities(self):
        """       
        use:
            - calculate weekly price elasticities for whole dataset

        input:
            - baskets: pd.DataFrame
            - coupons: pd.DataFrame

        return: pd.DataFrame
            - elasticities df
        """
        try:
            elasticities = pd.read_parquet('../data/elasticities.parquet', engine = 'pyarrow')
            print("Read Elasticities.parquet from disk...")
            return elasticities
        except:
            print("Elasticities will be generated...")
        
            baskets = self.baskets
            coupons = self.coupons

            start = time.time()
            elasticities = pd.DataFrame()
            total_basket_count = 100000

            for i in baskets['week'].unique():
                basket_week = baskets[baskets['week'] == i].copy()
                coupons_week = coupons[coupons['week'] == i].copy()
                basket_week = basket_week.merge(coupons_week, how = "outer")
                basket_week['discount'] = basket_week['discount'].fillna(0)
                basket_week['price'] = basket_week['price'].fillna(0)
                elast = []
                temp = pd.DataFrame()
                temp['product'] = np.arange(250)
                temp['week'] = i + 1

                for i in range(250):
                    reg_price = basket_week[basket_week['product'] == i]
                    reg_price_buy = len(reg_price[reg_price['discount'] == 0])
                    all_discounts_offers = len(reg_price[reg_price['discount'] > 0])
                    reg_price_offer = total_basket_count - all_discounts_offers
                    reg_price_buy_rate = reg_price_buy / reg_price_offer
                    discount_30 = reg_price[reg_price['discount'] == 30]
                    discount_30_offer = len(discount_30)
                    discount_30_buy = (discount_30['price'] != 0).sum()
                    discount_buy_rate = discount_30_buy / discount_30_offer
                    elast.append((discount_buy_rate - reg_price_buy_rate) / (0.3 * reg_price_buy_rate))
                temp['elast'] = elast
                elasticities = elasticities.append(temp, ignore_index = True)

            end = time.time()
            del baskets, coupons
            print("Elapsed time: ", end - start)
            elasticities.to_parquet('../data/elasticities.parquet', engine = 'pyarrow')
            print("Elasticities.parquet has been saved to disk")
            return elasticities