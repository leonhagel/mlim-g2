import pandas as pd
import Utils
import itertools


class DataLoader:
    
    def __init__(self, config):
        self.expected_input = ['baskets', 'coupons', 'coupons_index']
        self.config = config
        self.get_dataset(config)
    

    def get_dataset(self, config):
        dataset = Utils.parquet_loader(
            parquet_name = "dataset",
            path = config['data']['path'],
            callback = self.create_dataset
        )
        self.dataset = dataset
        return dataset
        

    def create_dataset(self):
        baskets_coupons = self.merge_baskets_coupons()
        data = self.clean(baskets_coupons)      
        week_hist = self.get_week_hist(data)
        last_week_mode_price = self.get_last_week_mode_price(data)

        dataset = self.build_dataset_from_config(data)
        dataset = dataset.merge(week_hist, how="left")
        dataset = dataset.merge(last_week_mode_price, how="left") #on=['product', 'week']
        dataset = self.impute_missing_prices(dataset)
        return dataset
        

    def reduce_by_shopper(self, data):
        max_shoppers = self.config['model']['n_shoppers']
        reduced_data = data[data['shopper'] < max_shoppers]
        return reduced_data
        #return data
    

    def merge_baskets_coupons(self):
        inputs = {name: self.read(name) for name in self.expected_input}
        self.inputs = inputs
        baskets = self.reduce_by_shopper(inputs['baskets'])
        coupons = self.reduce_by_shopper(inputs['coupons'])        
        baskets_coupons = baskets.merge(coupons, how="outer")
        return baskets_coupons
        
        
    def read(self, name):
        data_config = self.config['data']
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
        #data = self.reduce_by_shopper(data)
        return data


    def get_week_hist(self, data):
        '''
        Get list of purchase weeks for all product-shopper combinations
        '''
        purchases = data[data["purchased"] == 1]
        week_hist = purchases.groupby(['product', 'shopper'])['week'].apply(list).reset_index(name='week_hist')
        return week_hist
 
    
    def get_last_week_mode_price(self, dataset):
        '''
        Get mode price of prev week for every product-week combination
        (This will set the first week to NaN for every product)
        '''
        get_mode = lambda x: pd.Series.mode(x)[0]

        price_data = dataset.groupby(['product', 'week']).agg(
            week_mode_price=('price', get_mode)
        ).reset_index()

        price_data['last_week_mode_price'] = price_data.groupby('product')['week_mode_price'].shift()
        price_data = price_data.drop(columns=['week_mode_price'])
        return price_data
    
 
    def impute_missing_prices(self, dataset):
        '''
        Replace missing product prices with the mode price of the previous week
        '''
        dataset['price'] = dataset['price'].fillna(dataset['last_week_mode_price'])
        dataset = dataset.drop(columns=['last_week_mode_price'])
        print("replaced missing prices with mode")
        return dataset


    def build_dataset_from_config(self, data):
        model_config = self.config['model']
        '''
        Initialize dataset based on configured test week and train_window
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