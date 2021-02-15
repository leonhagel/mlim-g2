## super(): Helper
class Helper:
    
    def __init__(self):
        self.data = {}
        self.mappings = {}
        
    def _load(self, filepath):
        import pandas as pd
        file_type = filepath.split('.')[1]
        
        if file_type == 'parquet':
            output = pd.read_parquet(filepath)
        
        return output

    
    # internal functions
    def __getitem__(self, item):
        return eval(f"self.{item}")
    
    
    def __setitem__(self, item, value):
        exec(f"self.{item} = value")
        
    
    def _format_time(self, seconds):
        return '{:02.0f}'.format(seconds//60)+':'+'{:02.0f}'.format(seconds%60)
    
    
    
    # loading from disk
    def load(self, files:dict):
        for attr, files in files.items():
            if type(files) == list:
                for file in files:
                    name = file.split('.')[0].split('/')[-1]
                    self[attr][name] = self._load(file)
            else:
                self[attr] = self._load(files)
    
    
    def _dump_data(self, export_path, name):
        self.data[name].to_parquet(f"{export_path}{name}.parquet")
        
    
    def _dump_map(self, export_path, name):
        self.mappings[name].to_parquet(f"{export_path}{name}.parquet")
    
    
    def dump(self, export_path, which='all'):
        if which == 'all':
            # data
            provided_data = ['baskets', 'coupon_index', 'coupons']
            data = [data for data in self.data.keys() if data not in provided_data]
            
            # mappings
            mappings = list(self.mappings.keys())
            
            # other attributes and objects
            other = []
            
            which = data + mappings + other
        
        if which == 'data':
            provided_data = ['baskets', 'coupon_index', 'coupons']
            which = [data for data in self.data.keys() if data not in provided_data]
        
        if which == 'mappings':
            which = list(self.mappings.keys())
        
        for name in which:
            if name in self.data.keys():
                self._dump_data(export_path, name)
            if name in self.mappings.keys():
                self._dump_map(export_path, name)
        
    # data preparation
    def get_merged(self, drop=False):
        similar = [x for x in self.data['coupons'].columns if x in self.data['baskets'].columns] # i.e. ['shopper', 'product', 'week'] 
        self.data['merged'] = self.data['baskets'].merge(self.data['coupons'], how='outer', left_on=similar, right_on=similar)
        if drop:
            self.data.pop('baskets')
            self.data.pop('coupons')    

            
    def reduce_data_size(self, df):
        import numpy as np
        
        max_integer_values = {127: 'int8', 32767: 'int16', 2147483647: 'int32'}
        for column, dtype in df.dtypes.items():    
            if np.issubdtype(dtype, np.integer):
                # determining the minimum dtype
                max_value = np.max([abs(df[column].min()), df[column].max()])
                max_array = np.array(list(max_integer_values.keys()))
                max_idx = max_array[max_array > max_value][0]
                # converting integers
                df[column] = df[column].astype(max_integer_values[max_idx])
            # converting float
            if np.issubdtype(dtype, np.floating):
                df[column] = df[column].astype('float32')
        return df
    
    
    def reduce_shopper(self, df, shopper_range:tuple=(0,1999)):
        lower = df['shopper'] >= shopper_range[0]
        upper = df['shopper'] <= shopper_range[1]
        return df.loc[lower & upper]
    
    
    def clean(self, df='merged', shopper_range=(0,1999)):
        df = self.data['merged'].copy() if list(df) == list('merged') else df
        df = self.reduce_shopper(df, shopper_range)
    
        # cleaning
        df['discount'].fillna(0, inplace=True)
        df['discount'] = df['discount'] / 100
        df['price'] = df['price'] / (1 - df['discount'])
        # target
        df['purchased'] = df['price'].notna().astype('int8')    
        return df
    
    
    # data mappings
    def _init_df_map(self, rows, columns, initial_array=[]):
        from copy import deepcopy
        import pandas as pd
        
        rows = range(rows[0], rows[1]+1) if type(rows) == tuple else rows
        columns = range(columns[0], columns[1]+1) if type(columns) == tuple else columns

        rows = {row: deepcopy(initial_array) for row in rows}
        return pd.DataFrame({str(column): deepcopy(rows) for column in columns})
    
    
    def _get_mapping(self, df, row_name, column_name, value_name, rows='all', columns='all', initial_array=[]):
        from tqdm import tqdm
        tqdm.pandas()
        
        rows = (df[row_name].min(), df[row_name].max())
        columns = (df[column_name].min(), df[column_name].max())
        mapping = self._init_df_map(rows=rows, columns=columns, initial_array=initial_array)
        df.progress_apply(lambda row: mapping.loc[int(row[row_name]), str(int(row[column_name]))].append(row[value_name]), axis = 1)
        return mapping
    
    
    def get_mappings(self, config:dict):
        for name, cnfg in config.items():
            self.mappings[name] = self._get_mapping(**cnfg)

    
    def aggregate_price_map(self, aggregation_function='mode', verbose=1):
        from tqdm import tqdm
        import pandas as pd
        
        if aggregation_function == 'mode':
            import scipy.stats
            aggregation_function = lambda array: scipy.stats.mode(array)[0][0]
    
        price_map = pd.DataFrame()
        for column in tqdm(self.mappings['prices'].columns) if verbose >=1 else self.mappings['prices'].columns:
            price_map[column] = self.mappings['prices'][column].apply(aggregation_function)
    
        return price_map

    
# Purchase porbabilities
class Purchase_Probabilities(Helper):
    
    def __init__(self):
        super().__init__()
        
    
    # creating purchase history features
    def get_history(self, shopper, product, week):
        import numpy as np
        arr = np.array(self.mappings['product_histories'].loc[shopper, str(product)])
        return arr[arr < week]

    
    def get_last_purchase(self, shopper, product, week):
        return self.get_history(shopper, product, week)[-1]

    
    def get_trend(self, shopper, product, week, trend_window): # i.e. purchase frequency over the specified trend window
        import numpy as np
        history = self.get_history(shopper, product, week)
        return np.unique(history[history >= week - trend_window]).shape[0] / trend_window


    def prepare(self, df='clean', shopper=(0,1999), week=(0,89), product=(0,249), price_aggregation_fn='mode'):
        import numpy as np
        import pandas as pd
        import itertools
        import time
        from tqdm import tqdm
        tqdm.pandas()
        
    
        start = time.time()
        if type(df) == str:
            df = self.data[df]

        shopper = range(shopper[0], shopper[-1]+1) if type(shopper) != list else shopper
        week = range(week[0], week[-1]+1) if type(week) != list else week
        product = range(product[0], product[-1]+1) if type(product) != list else product

        print(f"[prepare] itertools... (elapsed time: {self._format_time(time.time()-start)})")
        output = pd.DataFrame(itertools.product(shopper, week, product))
        output.rename(columns={0:'shopper', 1:'week', 2:'product'}, inplace=True)
        print(f"[prepare] merge... (elapsed time: {self._format_time(time.time()-start)})")
        if all([type(df) != pd.core.frame.DataFrame, df == None]):
            output['price'] = None
            output['discount'] = None
            output['purchased'] = None
        else:
            output = output.merge(df, how='left', left_on=list(output.columns), right_on=list(output.columns))
            output['purchased'].fillna(0, inplace=True)
            output['discount'].fillna(0, inplace=True)


        print(f"[prepare] cleaning... (elapsed time: {self._format_time(time.time()-start)})")
        price_map = self.aggregate_price_map(price_aggregation_fn, verbose = 0)
        output.loc[output['price'].isna(), 'price'] = output.loc[output['price'].isna(), :].progress_apply(lambda row: price_map.loc[row['week']-1, str(int(row['product']))], axis=1)

        print(f"[prepare] feature creation... (elapsed time: {self._format_time(time.time()-start)})")
        # last purchase
        output['weeks_since_last_purchase'] = output.progress_apply(lambda row: self.get_last_purchase(int(row['shopper']), str(int(row['product'])), row['week']), axis=1)
        output['weeks_since_last_purchase'] = output['week'] - output['weeks_since_last_purchase']
        output.loc[output['weeks_since_last_purchase'] == np.inf, 'weeks_since_last_purchase'] = np.ceil(output['week'].max() * 1.15)
        # trends
        trend_windows = [1, 3, 5]
        for window in trend_windows:
            output['trend_'+str(window)] = output.progress_apply(lambda row: self.get_trend(int(row['shopper']), str(int(row['product'])), row['week'], window), axis=1)
        output['product_freq'] = output.progress_apply(lambda row: self.get_trend(int(row['shopper']), str(int(row['product'])), row['week'], row['week']), axis=1)

        print(f"[prepare] done (elapsed time: {self._format_time(time.time()-start)})")
        return output
        
        
    def train_test_split(self, test_week, train_window, df='purchase', features='default'):
        import category_encoders as ce
        from IPython.display import clear_output
        if type(df) == str:
            df = self.data[df]
        start = test_week - train_window
    
        train = df.loc[(df['week'] >= start) & (df['week'] < test_week), :]
        test = df.loc[(df['week'] == test_week), :]
    
        # WOE category encoding
        encoder = ce.WOEEncoder()
        train.loc[:,'shopper_WOE'] = encoder.fit_transform(train['shopper'].astype('category'), train['purchased'])['shopper'].values
        test.loc[:,'shopper_WOE'] = encoder.transform(test['shopper'].astype('category'))['shopper'].values
        encoder = ce.WOEEncoder()
        train.loc[:,'product_WOE'] = encoder.fit_transform(train['product'].astype('category'), train['purchased'])['product'].values
        test.loc[:,'product_WOE'] = encoder.transform(test['product'].astype('category'))['product'].values
        clear_output()
    
        # final split
        features = [col for col in train.columns if col not in ['purchased', 'shopper', 'week', 'product', 'product_history', 'last_purchase']] if features == 'default' else features
        X_train = train.loc[:, features]
        y_train = train['purchased']
        X_test = test.loc[:, features]
        y_test = test['purchased']
    
        return X_train, y_train, X_test, y_test
    
    
    def _fit_lgbm(self, X_train, y_train, **kwargs):
        import lightgbm
        model = lightgbm.LGBMClassifier()
        model.fit(X_train, y_train)
        return model

    
    def fit(self, model_type, X_train, y_train, **kwargs):
        return eval(f"self._fit_{model_type}(X_train, y_train, **kwargs)")
    
    
    def predict(self, model, X):
        import lightgbm
        if type(model) == lightgbm.sklearn.LGBMClassifier:
            y_hat = model.predict_proba(X)[:, 1]
        return y_hat
    
    
    def score(self, y_true, y_hat, metric='log_loss'):
        if metric == 'log_loss':
            import sklearn
            metric = sklearn.metrics.log_loss
    
        return metric(y_true, y_hat)