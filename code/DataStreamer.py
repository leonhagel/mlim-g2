import pandas as pd
import Utils
import itertools


class DataStreamer:
    
    def __init__(self, config):
        self.expected_input = ['baskets', 'coupons', 'coupons_index']
        self.config = config
        self.load(config)
    
    def load(self, config):
        baskets_coupons = Utils.parquet_loader(
            parquet_name = "baskets_coupons",
            path = config['data']['path'],
            callback = self.load_data
        )
        
        data = self.clean(baskets_coupons)      
        week_hist = self.get_week_hist(data)
        dataset = self.create_dataset(data, config['model'])
        
        self.dataset = dataset.merge(week_hist, how="left")
        print('dataset is ready!') 
        return self.dataset

        
    
    def load_data(self):
        data_config = self.config['data']
        inputs = {name: self.read(name, data_config) for name in self.expected_input}
        baskets_merged = inputs["baskets"].merge(inputs["coupons"], how="outer")
        self.inputs = inputs
        return baskets_merged
        
        
    def read(self, name, data_config):
        path = data_config['path']
        filename = data_config['files'][name]
        data = pd.read_parquet(path + filename)
        data_compressed = Utils.reduce_mem_usage(filename, data)
        return data_compressed

        
    def clean(self, data):
        data["discount"].fillna(0, inplace=True)
        data["discount"] = data["discount"] / 100
        data["price"] = data["price"] / (1 - data["discount"])
        data["purchased"] = data["price"].notna().astype("int8")
        

        #read this from config
        # todo: reduce before merging makes more sense
        # -----------------------------------------
        max_shoppers = 2000
        data = data[data['shopper'] < max_shoppers]
        # -----------------------------------------
        return data


    def get_week_hist(self, data):

        n_rows = data["shopper"].nunique()
        n_cols = data["product"].nunique()
        
        table = pd.DataFrame(itertools.product(list(range(n_rows)), list(range(n_cols))))
        table.columns = ['shopper', "product"]
        
        data_purchased = data[data["purchased"] == 1]
        hist = data_purchased.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='week_hist')
        week_hist = table.merge(hist, how='left')
        
        return week_hist

    
    def get_mode_prices(self, dataset):
        """
        returns mode price for every product-week combination 
        table columns: product, week, mode_price
        """
        df = dataset.copy()
        df = df[df["price"].notna()]

        get_mode = lambda x: pd.Series.mode(x)[0]
        
        # todo: don't use future data, group by week
        mode_prices = df.groupby('product').agg(
            mode_price=('price', get_mode)
        ).reset_index()

        return mode_prices
    
 
    # replace missing prices with mode price in associated week
    # ------------------------------------------------------------------------------  
    def impute_missing_prices(self, dataset): 
        mode_prices = self.get_mode_prices(dataset)
        dataset = dataset.merge(mode_prices, how='left', on=['product'])
        dataset['price'] = dataset['price'].fillna(dataset['mode_price'])
        dataset.drop('mode_price', axis=1, inplace=True)
        print("replaced missing prices with mode")
        
        return dataset


    def create_dataset(self, data, config):
        
        start_week = config['test_week'] - config['train_window']
        end_week = config['test_week'] + 1
        
        weeks = range(start_week, end_week)
        shoppers = range(config['n_shoppers'])
        products = range(config['n_products'])
        
        dataset = pd.DataFrame(itertools.product(weeks, shoppers, products))
        dataset.columns = ['week', 'shopper', 'product']        
        dataset = dataset.merge(data, how="left")
        dataset["purchased"].fillna(0, inplace=True)
        dataset["discount"].fillna(0, inplace=True)

        dataset = self.impute_missing_prices(dataset)
        
        
        return dataset
    
    
