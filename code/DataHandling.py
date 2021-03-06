import pandas as pd
import Utils


class DataLoader:
    
    def __init__(self, config):
        self.config = config
        self.load(config)


    def load(self, config):
        data_path = config['path']
        output = {}
        for name, filename in config['files'].items():
            data = pd.read_parquet(data_path + filename)
            output[name] = Utils.reduce_mem_usage(filename, data)
        self.data = output
        
        def merge_baskets_and_coupons():
            return output["baskets"].merge(output["coupons"], how="outer")
        
        self.data["merged"] = Utils.parquet_loader(
            parquet_name = "merged",
            path = data_path,
            callback = merge_baskets_and_coupons
        )