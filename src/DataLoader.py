import itertools
import pandas as pd
import numpy as np
import Utils


class DataLoader:
    
    def __init__(self, config):
        self.expected_input = ['baskets', 'coupons', 'coupon_index']
        self.config = config
        

    def get_dataset(self):
        '''
        If available, read dataset.parquet.gzip from cache
        Ohterwise create dataset and write it to cache
        '''
        dataset = Utils.parquet_caching(
            parquet_name = "dataset",
            callback = self.create_dataset
        )
        self.dataset = dataset
        return dataset
        
        
    def create_dataset(self):
        baskets_coupons = self.merge_baskets_coupons()
        data = self.clean(baskets_coupons)      
        week_hist = self.get_week_hist(data)
        discount_received_weeks = self.get_discount_received_weeks(data)
        discount_redeemed_weeks = self.get_discount_redeemed_weeks(data)
        last_week_mode_price = self.get_last_week_mode_price(data)
        dataset = self.build_dataset_from_config(data)
        dataset = dataset.merge(week_hist, how="left")
        dataset = dataset.merge(discount_received_weeks, how="left")
        dataset = dataset.merge(discount_redeemed_weeks, how="left")
        dataset = dataset.merge(last_week_mode_price, how="left")
        dataset = self.impute_missing_prices(dataset)
        elasticities = self.get_elasticities()
        #redemption_rate, costumer_redemption_rate, discount_buy = self.get_coupon_rates(baskets_coupons)
        #dataset = dataset.merge(elasticities, how="left")
        #dataset = dataset.merge(redemption_rate, how="left", on='product')
        #dataset = dataset.merge(costumer_redemption_rate, how="left", on=['shopper', 'product'])
        #dataset = dataset.merge(discount_buy, how="left", on=['shopper', 'product'])
        return dataset
        

    def reduce_by_shopper(self, data):
        '''
        Reduce data to first n shoppers specified in model config
        '''
        max_shoppers = self.config['model']['n_shoppers']
        reduced_data = data[data['shopper'] < max_shoppers]
        return reduced_data
    

    def merge_baskets_coupons(self):
        inputs = {name: self.read(name) for name in self.expected_input}
        self.inputs = inputs
        baskets = self.reduce_by_shopper(inputs['baskets'])
        coupons = self.reduce_by_shopper(inputs['coupons'])        
        baskets_coupons = baskets.merge(coupons, how="outer")
        return baskets_coupons
        
        
    def read(self, name):
        '''
        Read memory reduced parquet files
        '''
        path = self.config['input']['path']
        filename = self.config['input']['files'][name]
        data = pd.read_parquet(path + filename)
        data_compressed = Utils.reduce_mem_usage(filename, data)
        return data_compressed

        
    def clean(self, data):
        '''
        Basic data cleaning after baskets-coupons merge
        '''
        data["discount"].fillna(0, inplace=True)
        data["discount"] = data["discount"] / 100
        data["price"] = data["price"] / (1 - data["discount"])
        data["purchased"] = data["price"].notna().astype("int8")
        return data


    def get_week_hist(self, data):
        '''
        Helper to derive time based features later on
        computes list of purchase weeks for all product-shopper combinations
        '''
        purchases = data[data["purchased"] == 1]
        week_hist = purchases.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='week_hist')
        return week_hist
 

    def get_discount_received_weeks(self, data):
        '''
        Helper to derive coupon related features later on
        computes list of coupon received weeks for all product-shopper combinations
        '''
        discount_received = data[data["discount"] != 0]
        discount_received_weeks = discount_received.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='discount_received_weeks')
        return discount_received_weeks

    
    def get_discount_redeemed_weeks(self, data):
        '''
        Helper to derive coupon related features later on
        computes list of coupon redemmed weeks for all product-shopper combinations
        '''
        discount_redeemed = data[(data["purchased"] == 1) & (data["discount"] != 0)]
        discount_redeemed_weeks = discount_redeemed.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='discount_redeemed_weeks')
        return discount_redeemed_weeks
 
    
    def get_last_week_mode_price(self, dataset):
        '''
        Get mode price of prev week for every product-week combination
        (This will set the first week to NaN for every product)
        '''
        get_mode = lambda x: pd.Series.mode(x)[0]

        price_data = dataset.groupby(['product', 'week']).agg(
            last_week_mode_price=('price', get_mode)
        ).reset_index()
        price_data['week'] = price_data['week'] + 1
        return price_data
    
 
    def impute_missing_prices(self, dataset):
        '''
        Replace missing prices with the mode price of the previous week
        '''
        dataset['price'] = dataset['price'].fillna(dataset['last_week_mode_price'])
        dataset = dataset.drop(columns=['last_week_mode_price'])
        print("replaced missing prices with mode")
        return dataset


    def build_dataset_from_config(self, data):
        model_config = self.config['model']
        '''
        Build dataset based on configured test week and train_window
        '''
        start_week = model_config['test_week'] - model_config['train_window']
        end_week = model_config['test_week'] + 1
        
        weeks = range(start_week, end_week)
        shoppers = range(model_config['n_shoppers'])
        products = range(model_config['n_products'])
        
        dataset = pd.DataFrame(itertools.product(weeks, shoppers, products))
        dataset.columns = ['week', 'shopper', 'product']        
        dataset = dataset.merge(data, how="left")
        dataset["purchased"].fillna(0, inplace=True)
        dataset["discount"].fillna(0, inplace=True)
        return dataset
    
   ## benedikt  
    
    def get_coupon_rates(self, data):
        '''
        Calculate costumer and product specific redemption likeliehoods
        and calculate if a costumer buys a product only because of coupons
        uses all historic data prior to train window
        '''
        test_week = self.config['model']['test_week']
        train_window = self.config['model']['train_window']
        week_limit = test_week - train_window
        
        data = data[data['week'] < week_limit]
        coupons = data[data['discount'] > 0].copy()
        
        # how likely is a product specific coupon to be redeemed
        redemption_rate = coupons.groupby('product')['purchased'].mean()
        redemption_rate = redemption_rate.rename('redemption_rate')

        # how likely is a specific costumer to redeem a coupon for a specific product
        costumer_redemption_rate = coupons.groupby(['shopper', 'product'])['purchased'].mean()
        costumer_redemption_rate = costumer_redemption_rate.rename('costumer_redemption_rate')

        # did the costumer buy the product only because of the discount
        buy_all = data.groupby(['shopper', 'product']).size()
        discount = coupons.groupby(['shopper', 'product']).size()
        discount_buy = discount / buy_all
        discount_buy = discount_buy.fillna(0)
        discount_buy = discount_buy.rename('discount_buy')
        
        return redemption_rate, costumer_redemption_rate, discount_buy
        
        
    def get_elasticities(self):
        '''
        Calculate week and product specific price elasticities using all shoppers
        '''
        elasticities = pd.DataFrame()
        total_basket_count = 100000
        end_week = self.config['model']['test_week']
        start_week = end_week - self.config['model']['train_window']
        
        baskets = self.inputs['baskets']
        coupons = self.inputs['coupons']
        
        baskets = baskets[baskets['week'] >= start_week -1] 
        baskets = baskets[baskets['week'] <= end_week]
        coupons = coupons[coupons['week'] >= start_week -1] 
        coupons = coupons[coupons['week'] <= end_week]
        baskets_coupons = baskets.merge(coupons, how = "outer")

        baskets_coupons['discount'] = baskets_coupons['discount'].fillna(0)
        baskets_coupons['price'] = baskets_coupons['price'].fillna(0)
        
        for i in baskets_coupons['week'].unique():
            basket_week = baskets_coupons[baskets_coupons['week'] == i].copy()
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
        del baskets, coupons, baskets_coupons
        return elasticities
