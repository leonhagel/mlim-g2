import pandas as pd
import Utils
import itertools


class DataLoader:
    
    def __init__(self, config):
        self.expected_input = ['baskets', 'coupons', 'coupons_index']
        self.config = config
        self.load(config)
    
    
    def read(self, config, name):
        path = config['path']
        filename = config['files'][name]
        data = pd.read_parquet(path + filename)
        data_compressed = Utils.reduce_mem_usage(filename, data)
        return data_compressed
    
    
    def load_data(self):
        inputs = {name: self.read(self.config, name) for name in self.expected_input}
        self.data = inputs["baskets"].merge(inputs["coupons"], how="outer")
        self.inputs = inputs

        
    def clean(self, data):
        data["discount"].fillna(0, inplace=True)
        data["discount"] = data["discount"] / 100
        data["price"] = data["price"] / (1 - data["discount"])
        data["purchased"] = data["price"].notna().astype("int8")
        
        # comment this out for production
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


    def get_output(self, data):
        # read this from config FILE
        n_shoppers = 2000
        n_products = 250
        shopper = range(n_shoppers)
        product = range(n_products)
        test_week = 90
        train_window = 4
        week = range(test_week - train_window, test_week + 1)

        #trend_windows = [1, 3, 5]
        
        output = pd.DataFrame(itertools.product(shopper, week, product))
        output.columns = ['shopper', 'week', 'product']        
        output = output.merge(data, how="left")
        output["purchased"].fillna(0, inplace=True)
        output["discount"].fillna(0, inplace=True)
        
        return output
    
    
    def load(self, config):
        data = Utils.parquet_loader(
            parquet_name = "merged",
            path = config['path'],
            callback = self.load_data
        )
        
        data = self.clean(data)
        
        self.data = data
        self.week_hist = self.get_week_hist(data)
        self.output = self.get_output(data)
        print('everything loaded!')
        

        

        

 
        