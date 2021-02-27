# class inheritance level: 0 - helper
# ==================================================================================
class Helper:
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        0

    goal:
        reuse code efficiently

    functionality:
        - load data/exports from disk (.load)
        - store/export data to disk (.dump)
        - generic data preparation (.get_merged, .clean)
        - initialize and create mappings (.get_mappings)
        - (convenience functions: dict-like behaviour, format time)
    """

    def __init__(self):

        """
        attributes:
            - data: dict
                - contains the loaded data sets
                - if data set has been loaded using the load method:
                  key = filename (w/o filetype)
            - mappings: dict
                - contains mappings created using the get_mappings method

        public methods:
            - load: reads files from disk
            - dump: stores objects to disk
            - reduce_data_size: reduces memory usage of dataframes
            - get_merged: merges the provided data
            - reduce_shoppers: reduces the shoppers in a dataframe to a specified
              shopper range
            - clean: generic data preparation
            - get_mappings: creates a data map according to the specified configuration
        """

        self.data = {}
        self.mappings = {}

    # load data/exports from disk
    # ----------------------------------------------------------------------------------
    def load(self, files: dict):
        """
        use:
            - load files from the files-dictionary and store at correct location

        inputs:
            - files: dict
                - key: str -> location/attribute where the data should be stored
                  (e.g. data, mapping)
                - value: list or str --> filepath or list of filepaths

        return: None
        """

        for attr, files in files.items():
            if type(files) == list:
                for file in files:
                    name = file.split(".")[0].split("/")[-1]
                    self[attr][name] = self._load(file)
            else:
                self[attr] = self._load(files)

    def _load(self, filepath: str):
        """
        use:
            - actual reading operation

        supported filetypes: -> expandable using further if-clauses
            - parquet

        input:
            - filepath: str

        return:
            - parquet: pd.DataFrame
        """

        import pandas as pd

        file_type = filepath.split(".")[1]

        if file_type == "parquet":
            output = pd.read_parquet(filepath)

        return output

    # store/export data to disk
    # ----------------------------------------------------------------------------------
    def dump(self, export_path: str, which: str = "all"):
        """
        use:
            - stores the specified object/data to the disk

        input:
            - export_path: str
                - path to the directory, where the objects should be stored
            - which: str
                - specifies which objects should be stored to disk
                - supported arguments:
                    - 'data': stores each object in the data dictionary
                    - 'mappings': stores each object in the mappings dictionary
                    - 'all': stores all supported object

        return: None
        """
        if which == "all":
            # data
            provided_data = ["baskets", "coupon_index", "coupons"]
            data = [data for data in self.data.keys() if data not in provided_data]
            # mappings
            mappings = list(self.mappings.keys())
            # other attributes and objects
            which = data + mappings
        elif which == "data":
            provided_data = ["baskets", "coupon_index", "coupons"]
            which = [data for data in self.data.keys() if data not in provided_data]
        elif which == "mappings":
            which = list(self.mappings.keys())
        else:
            raise ValueError(
                "'which'-argument not supported! Supported arguments: 'all', 'data', mappings"
            )

        for name in which:
            if name in self.data.keys():
                self._dump_data(export_path, name)
            if name in self.mappings.keys():
                self._dump_map(export_path, name)

    def _dump_data(self, export_path: str, name: str):
        """
        use:
            - stores 'data' objects to the export path

        input:
            - export_path: str
                - export path provided by the .load method
            - name: str
                - filename for the object to be stored

        return: None
        """
        self.data[name].to_parquet(f"{export_path}{name}.parquet")

    def _dump_map(self, export_path: str, name: str):
        """
        use:
            - stores 'mappings' objects to the export path

        input:
            - export_path: str
                - export path provided by the .load method
            - name: str
                - filename for the object to be stored

        return: None
        """
        self.mappings[name].to_parquet(f"{export_path}{name}.parquet")

    # generic data preparation
    # ----------------------------------------------------------------------------------
    def reduce_data_size(self, df):
        """
        use:
            - reduces memory usage of dataframes by converting integers to the lowest
              possible memory usage and float to float32

        input:
            - df: pd.DataFrame
                - dataframe with high memory usage

        return: pd.DataFrame
            - dataframe with lower memory usage
        """

        import numpy as np

        max_integer_values = {127: "int8", 32767: "int16", 2147483647: "int32"}
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
                df[column] = df[column].astype("float32")
        return df

    def get_merged(self, drop: bool = False):
        """
        use:
            - merge the provided dataframes coupons.parquet, baskets.parquet

        requirement:
            - data needs to be loaded to the 'data' dictionary
            - naming needs to be 'coupons', 'baskets'
            - Hint: load data using .load(); use filenames coupons.parquet, baskets.parquet

        input:
            - drop=False:
                - indicates whether input dataframes should be dropped, the merged data
                  will be kept

        return: pd.DataFrame
            - merged dataframe
        """
        try:
            self.data["merged"]
        except KeyError:
            similar = [
                x
                for x in self.data["coupons"].columns
                if x in self.data["baskets"].columns
            ]  # i.e. ['shopper', 'product', 'week']
            self.data["merged"] = self.data["baskets"].merge(
                self.data["coupons"], how="outer", left_on=similar, right_on=similar
            )
        if drop:
            self.data.pop("baskets")
            self.data.pop("coupons")
        return self.data["merged"]

    def reduce_shopper(self, df, shopper_range: tuple = (0, 1999)):
        """
        use:
            - reduces the shoppers to the specified id range

        input:
            - df: pd.DataFrame
                - dataframe for which the shoppers should be reduced
            - shopper_range: tuple
                - first and last id for the range of shoppers, which should be included
                  in the resulting dataframe

        return: pd.DataFrame
            - input dataframe, reduced to only the shoppers specified in the shopper range

        """

        lower = df["shopper"] >= shopper_range[0]
        upper = df["shopper"] <= shopper_range[1]
        return df.loc[lower & upper]

    def clean(self, df="merged", shopper_range: tuple = (0, 1999)):
        """
        use:
            - performs generic data preparations

        cleaning operations:
            - shoppers: reduces to shopper range
            - discount: missing discount values will be set to 0
            - prices: calculate prices without discounts
            - target - purchased: create binary target based on whether the data point
              was included in the baskets data

        input:
            - df='merged': pd.DataFrame
                - dataframe which should be cleaned
                - default='merged': uses the merged data located at self.data['merged']
            - shopper_range: tuple
                - first and last id of a shopper range which should be included in the
                  cleaned data
                - default=(0,1999): reduces the shoppers to the range which is required
                  for the final prediction

        return: pd.DataFrame
            - cleaned and reduced dataframe
        """

        if list(df) == list("merged"):
            df = self.data["merged"].copy()
        # reducing shoppers
        df = self.reduce_shopper(df, shopper_range)
        # cleaning
        df["discount"].fillna(0, inplace=True)
        df["discount"] = df["discount"] / 100
        df["price"] = df["price"] / (1 - df["discount"])
        # target
        df["purchased"] = df["price"].notna().astype("int8")
        self.data["clean"] = df
        return df

    # mappings: initialize and create mappings
    # ----------------------------------------------------------------------------------
    def get_mappings(self, config: dict):
        """
        use:
            - creates mappings based on the provided configuration by looping over the
              data and storing the current value to the correct location in the map
            - stores the map to the 'mappings'-attribute

        input:
            - config: dict
                - key: name of the mapping to be created
                - values: kwargs:dict
                     - 'df': dataframe for which the map should be created
                     - 'row_name': name of the column in 'df' which will be the index
                       in the resulting dataframe/map
                     - 'column_name': name of the column in 'df' which will be the
                       column in the resulting dataframe/map
                     - 'value_name': name of the column in 'df' which is the feature of
                       interest. While iterating over the data, the value will be
                       appended to a list stored in the dataframe/map, located at
                       [row_name, column_name]
                     - 'initial_array': first elements in the value list
                         - default=[]: no first values

        return: pd.DataFrame
            - index: unique values of the 'row_name' column in 'df'
            - column: unique values of the 'column_name' column in 'df'
            - values: list of values for each index-column combination, based on the
              'value_name' column in 'df'
        """
        for name, cnfg in config.items():
            self.mappings[name] = self._get_mapping(**cnfg)

    def _init_df_map(self, rows, columns, initial_array: list = None):
        """
        use:
            - initialized empty map, using the specified row, columns and initial array

        input:
            - row: tuple or list
                - list or range which will specify the INDEX of the map
            - column: tuple or list
                - list or range which will specify the COLUMNS of the map
            - initial_array: list
                - first elements of the list. Will be applied to all row-column
                  combinations

        return: pd.DataFrame
            - empty dataframe map
        """
        from copy import deepcopy
        import pandas as pd

        if initial_array == None:
            initial_array = []

        # converting tuple to range
        if type(rows) == tuple:
            rows = range(rows[0], rows[1] + 1)
        if type(columns) == tuple:
            columns = range(columns[0], columns[1] + 1)

        rows = {row: deepcopy(initial_array) for row in rows}
        return pd.DataFrame({str(column): deepcopy(rows) for column in columns})

    def _get_mapping(
        self,
        df,
        row_name,
        column_name,
        value_name,
        initial_array: list = None,
    ):
        """
        use:
            - creates the the map by iterating over 'df'

        input:
             - 'df': dataframe for which the map should be created
             - 'row_name': name of the column in 'df' which will be the index in the
               resulting dataframe/map
             - 'column_name': name of the column in 'df' which will be the column in
               the resulting dataframe/map
             - 'value_name': name of the column in 'df' which is the feature of interest.
               While iterating over the data, the value will be appended to a list
               stored in the dataframe/map, located at [row_name, column_name]
             - 'initial_array': first elements in the value list
                 - default=[]: no first values

        return: pd.DataFrame
            - dataframe map
        """
        from tqdm import tqdm

        tqdm.pandas()

        if initial_array == None:
            initial_array = []

        rows = (df[row_name].min(), df[row_name].max())
        columns = (df[column_name].min(), df[column_name].max())
        mapping = self._init_df_map(
            rows=rows, columns=columns, initial_array=initial_array
        )
        df.progress_apply(
            lambda row: mapping.loc[
                int(row[row_name]), str(int(row[column_name]))
            ].append(row[value_name]),
            axis=1,
        )
        return mapping

    # convenience functions
    # ----------------------------------------------------------------------------------
    def __getitem__(self, item):
        """
        use:
            - dict-like behaviour: self['item'] <-> self.item
            - allows to get attributes using a str
        """
        return eval(f"self.{item}")

    def __setitem__(self, item, value):
        """
        use:
            - dict-like behaviour: self['item'] = value <-> self.item = value
            - allows to set attribute values using a str
        """
        exec(f"self.{item} = value")

    def _format_time(self, seconds):
        """
        use:
            - formats seconds into str time-format: 'mm:ss'
        """
        return "{:02.0f}".format(seconds // 60) + ":" + "{:02.0f}".format(seconds % 60)


# class inheritance level: 1 - Data preparation
# ==================================================================================
class Prices(Helper):
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        1

    goal:
        - cleaning the price feature by using an appropriate replacement approach for
          missing values

    functionality:
        - creating a price map which contains all prices of each product for each week
        - aggregate the list of prices to an appropriate single price
    """

    def __init__(self):
        """
        attributes:
            - (helper-attributes: data, mappings)

        public methods:
            - get_price_map: creates a price map for cleaning the price feature
            - aggregate_price_map: aggregates the list of prices to an appropriate
              replacement value for missing values

            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
        """
        super().__init__()

    # pipeline: creating the price map
    # ----------------------------------------------------------------------------------
    def get_price_map(self, df="merged"):
        # def pipeline_prices(self, df="merged"):
        """
        use:
            - creates the 'prices' map for cleaning the missing prices
            - stores the price map located at self.mappings['prices']
            - idea: replace missing prices with mode/mean of the product's prices in
              the current week to account for any price changes

        requirements:
            - df='merged': the merged data set needs to be located at self.data['merged']
                - Hint: merged data set can be created using self.get_merged()
        input:
            - df: pd.DataFrame
                - dataframe for which the price map should be created
                - default='merged': will take the 'merged' data to use full data w/o any
                  shopper reductions

        return: pd.DataFrame
            - 'prices' map for cleaning the prices (index: weeks, columns: products,
              values: list of the product's prices in the respective week)

        """
        import numpy as np

        # specifying the input data
        if type(df) == str:
            df = self.data[df].copy()

        df = df.loc[df["price"].notna(), :]

        # specifying the configuration for the map
        map_config = {
            "prices": {
                "df": df,
                "row_name": "week",
                "column_name": "product",
                "value_name": "price",
                "initial_array": [],
            }
        }
        self.get_mappings(map_config)
        return self.mappings["prices"]

    # aggregating the price map
    # ----------------------------------------------------------------------------------
    def aggregate_price_map(self, aggregation_function="mode", verbose=1):
        """
        use:
            - aggregates the list of prices for each product-week combination using
              the aggregation function

        requirements:
            - price map needs to be stored at self.mappings['prices']

        input:
            - aggregation_function: function(price_list)
                - function which aggregates the price list
                - default='mode': mode aggregation using scipy.stats.mode

        return: pd.DataFrame
            - aggregated price map containing product prices for each week
        """
        from tqdm import tqdm
        import pandas as pd

        if aggregation_function == "mode":
            import scipy.stats

            aggregation_function = lambda array: scipy.stats.mode(array)[0][0]

        price_map = pd.DataFrame()
        for column in (
            tqdm(self.mappings["prices"].columns)
            if verbose >= 1
            else self.mappings["prices"].columns
        ):
            price_map[column] = self.mappings["prices"][column].apply(
                aggregation_function
            )

        return price_map


class Product_Histories(Helper):
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        1

    goal:
        - creating purchase history features, e.g. for each shopper-product combination

    functionality:
        - creating a purchase history maps
        - aggregating the purchase history to extract further history-based features,
          e.g. purchase trends, time since last purchase, etc.
    """

    def __init__(self):
        """
        attributes:
            - (helper-attributes: data, mappings)

        public methods:
            - get_history_map: creates a purchase history map
            - get_history: returns the purchase history for a given point in time
            - get_last_purchase: returns the last purchase for a given point in time
            - get_trend: returns the purchase trend for a given trend window and
              point in time

            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
        """
        super().__init__()

    # creating purchase history maps
    # ----------------------------------------------------------------------------------
    def get_history_map(self, mapping: str = "product_histories"):
        #    def pipeline_histories(self, mapping: str = "product_histories"):
        """
        use:
            - creates the 'purchase_histories' map to be able to create history-based
              features
            - stores the purchase history map located at
              self.mappings['product_histories']

            *** @SASCHA: ***************************************************************
            - Possible to extend the function to purchase histories of product clusters
            - input: mapping='cluster_histories'
            - df: product clusters need to be added somehow
            - config: needs to be verified
            ****************************************************************************

        requirements:
            - data: requires either 'clean' or 'merged' data stored at self.data['clean']
              or self.data['merged'] respectively

        input:
            - mapping: str
                - name of the map to be created

        return: pd.DataFrame
            - 'product_histories' map to create further history-based features
              (index: shopper, columns: products, values: list of weeks in which each
              shopper purchased the respective product
        """

        import numpy as np

        # specifying the input data

        if mapping == "product_histories":
            try:
                df = self.data["clean"]
            except KeyError:
                df = self.clean()  # uses cleaned data only for shoppers 0 to 1999
        # if mapping == 'cluster_histories'
        # load df w/ product clusters

        # reduce to purchased items only
        df = df.loc[df["purchased"] == 1, :]

        # selecting the configuration for the map
        map_config = {
            "product_histories": {
                "df": df,
                "row_name": "shopper",
                "column_name": "product",
                "value_name": "week",
                "initial_array": [-np.inf],
            }  # ,
            # 'cluster_histories': {'df': df, 'row_name': 'shopper', 'column_name': '##CLUSTER_FEATURE_NAME##', 'value_name': 'week', 'initial_array': [-np.inf]}
        }
        map_config = {mapping: map_config[mapping]}
        self.get_mappings(map_config)
        return self.mappings[mapping]

    # aggregating the history map for feature creation
    # ----------------------------------------------------------------------------------
    def get_history(
        self, shopper: int, product: int, week: int, mapping: str = "product_histories"
    ):
        """
        use:
            - extracting the relevant purchase history given a shopper, product and week

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        input:
            - shopper: int
                - shopper id
            - product: int
                - product id
            - week: int
                - point in time for which the history should be returned
            - mapping: str
                - name of the relevant map stored at self.mappings[mapping]
                - default='product_histories': returns shopper-product histories

        return: np.array
            - 'product_histories': array which contains all week numbers of purchase
              weeks for the provided shopper-product combination and prior to the
              provided week
        """
        import numpy as np

        arr = np.array(self.mappings[mapping].loc[shopper, str(product)])
        return arr[arr < week]

    def get_last_purchase(
        self, shopper: int, product: int, week: int, mapping: str = "product_histories"
    ):
        """
        use:
            - receiving the last purchases week for the provided shopper-product
              combination at the provided point in time

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        input:
            - shopper: int
                - shopper id
            - product: int
                - product id
            - week: int
                - point in time for which the last purchase should be returned
            - mapping: str
                - name of the relevant map stored at self.mappings[mapping]
                - default='product_histories': returns shopper-product last purchases

        return: int
            - week of last purchase for the provided shopper-product combination and
              prior to the provided week
        """
        return self.get_history(shopper, product, week, mapping=mapping)[-1]

    def get_trend(
        self,
        shopper: int,
        product: int,
        week: int,
        trend_window: int,
        mapping: str = "product_histories",
    ):
        """
        use:
            - receiving a current purchase trend, i.e. purchase frequency over the
              specified trend window, for the provided shopper-product combination
              at the provided point in time

        requirements:
            - 'product_histories': map needs to be stored at
              self.mappings['product_histories']

        input:
            - shopper: int
                - shopper id
            - product: int
                - product id
            - week: int
                - point in time for which the last purchase should be returned
                        - mapping: str
            - trend_window: int
                - number of weeks prior to 'week' on which the trend calculations
            - mapping: str
                - name of the relevant map stored at self.mappings[mapping]
                - default='product_histories': returns shopper-product last purchases

        return: int
            - trend/purchase frequency for the provided trend window and
              shopper-product combination, prior to the provided week
        """
        import numpy as np

        history = self.get_history(shopper, product, week, mapping=mapping)
        return (
            np.unique(history[history >= week - trend_window]).shape[0] / trend_window
        )


class Redemptions(Helper):
    
    """    
        input: 
        week (touple, start, stop) 
        shopper (touple, start, stop)
        basket df variables: 'discount'(int/float), 'price'(int/float), 'product'(int), 'shopper'(int) (nan set to zero)
        
        output: basket df with 'product_redemption_likelihood', 'shopper_product_redemption_likelihood', 'discount_buy' added
    
    
    
    
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        - 1
    goal:
        - calculate costumer and product specific coupon redemption rates
    functionality:
        - take merged dataframe
        - calculate product and shopper_product coupon redemption rates
        - calculate products that have only been bought because of a discount    
    """
    
    def __init__(self):
        """
        attributes: 
            - ???

        public methods:
            - shopper_coupon_rates: resize input dataframe, calculate shopper_product_redemption_likelihood, discount_buy

            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
        """    
        super().__init__()
        
    def shopper_coupon_rates(self, basket, week, shopper):
         """       
        use:
            - merged and cleaned datafrage
            
        requirements:
            - dataframe defines as 'output'

        input:
            - product: int
                - product id
            - discount: float
                - discount rate
            - week: int
                - start and stop week for calculation
            - shopper: int
                - shopper id
            - price: float

        return: pd.DataFrame
            - features added
        """
        # data is shopper specific => use shoppers [0-1999] 
        # 0.1 sec for 1 week & 2000 shoppers    
        # data is very sparse if only one week is used 
            
        output = basket[(week[1] >= basket['week']) & (basket['week'] >= week[0])].copy()              # delete if specific weeks already as input
        output = output[(shopper[1] >= output['shopper']) & (output['shopper'] >= shopper[0])]         # delete if only 2000 shoppers as input
        coupons = output[output['discount'] > 0].copy()
        coupons.loc[(coupons['price'].notnull()), 'redeemed']  = 1
        coupons['redeemed'] = coupons['redeemed'].fillna(0)

        # coupon redemption likelihood by costumer & product
        costumer_redemption_rate = coupons.groupby(['shopper', 'product'])['redeemed'].mean()
        output = output.merge(costumer_redemption_rate, how = 'left', on = ['shopper', 'product'])
        output = output.rename(columns = {'redeemed' : 'shopper_product_redemption_likelihood'})


        # only bought because of discount?
        buy_all = output.groupby(['shopper', 'product']).size()
        discount = coupons.groupby(['shopper', 'product']).size()
        discount_buy = discount / buy_all
        discount_buy = discount_buy.fillna(0)
        discount_buy = discount_buy.rename('discount_buy')
        output = output.merge(discount_buy, how = 'left', on = ['shopper', 'product'])

        return output


class Elasticities(Helper):
    
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        - 1
    goal:
        - calculate week specific price elasticities
    functionality:
        - load and merge specified week from basket & coupons df
        - calculate product specific price elasticities from normal price & 30% discounts
    """
    
    def __init__(self):
        """
        attributes: 
            - ???

        public methods:
            - calc_week_elastics: pull and merge dataframes, return price elasticities

            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
        """    
        super().__init__()
        self.baskets = basket
        self.coupons = coupons

    
    def calc_week_elasticity(self, week):  # runtime of 3.3 sec
        """       
        use:
            - pull and merge dataframes, return price elasticities
            
        requirements:
            - 'basket.parquet' and 'coupons.parquet' needs to be loaded to class ???

        input:
            - product: int
                - product id
            - discount: float
                - discount rate
            - week: int
                - start and stop week for calculation
            - shopper: int
                - shopper id

        return: pd.DataFrame: float
            - price elasticity of each product
        """
        
        basket_week = basket[(week[1] >= basket['week']) & (basket['week'] >= week[0])].copy()
        coupons_week = coupons[(week[1] >= coupons['week']) & (coupons['week'] >= week[0])].copy()
        basket_week = basket_week.merge(coupons_week, how = "outer")
        
        basket_week['discount'] = basket_week['discount'].fillna(0)
        basket_week['price'] = basket_week['price'].fillna(0)

        elast = []
        total_basket_count = 100000 * (week[1] - week[0] + 1)

        for i in range(250):
            reg_price = basket_week[basket_week['product'] == i]
            reg_price_buy = len(reg_price[reg_price['discount'] == 0])
            all_discounts_offers = len(reg_price[reg_price['discount'] > 0])
            reg_price_offer = total_basket_count - all_discounts_offers
            reg_price_buy_rate = reg_price_buy / reg_price_offer
            discount_30 = reg_price[reg_price['discount'] == 0.3]
            discount_30_offer = len(discount_30)
            discount_30_buy = (discount_30['price'] != 0).sum()
            discount_buy_rate = discount_30_buy / discount_30_offer
            elast.append((discount_buy_rate - reg_price_buy_rate) / (0.3 * reg_price_buy_rate))

        elasticity_df = pd.DataFrame(elast)
        elasticity_df['product'] = elasticity_df.index 
        return elasticity_df
    
    
    
# Product clusters


# class inheritance level: 2 - Predicting purchase probabilities
# ==================================================================================
class Purchase_Probabilities(Product_Histories, Prices, Elasticities, Redemptions):
    """
    class inheritance level: (inheritance levels: 0 -> 1 -> 2 -> 3)
        2

    goal:
        - forecasting purchase probabilities

    functionality:
        - data preparation, i.e. final data cleaning, feature creation, feature selection,
          train test split
        - model training and evaluation
    """

    def __init__(self):
        """
        attributes:
            - shoppers: list
                - list of shoppers which are used for the model
            - test_week: int
                - test data: week for which the model should be tested
                - specified using self.train_test_split()
            - train_window: int
                - train data: number of week prior to the test week on which the model
                  will be trained
                - specified using self.train_test_split()
            - df: pd.DataFrame
                - input data for self.train_test_split()
                - specified using self.train_test_split()
            - features: list
                - names of features used for training the model
                - specified using self.train_test_split()
            - model_type: str
                - name of the model type used to specify which model should be used
                - specified using self.fit()
            - model: object
                - object of the trained model
                - specified using self.fit()

            - (helper-attributes: data, mappings)

        public methods:
            - prepare: prepares the data for model training - final data cleaning,
              feature creation
            - train_test_split: performs a time series split on the data + final
              preparation steps, feature selection
            - fit: trains the specified model
            - predict: predicts the specified model
            - score: calculates a model performance score

            - (prices: get_price_map, aggregate_price_map)
            - (product_histories: get_history_map, get_history, get_last_purchase,
              get_trend)
            - (Elasticities: calc_week_elasticity) 
            - (Redemptions: shopper_coupon_rates)
            - (helper: load, dump, reduce_data_size, get_merged_clean, get_mappings)
        """
        super().__init__()
        self.shoppers = None
        self.test_week = None
        self.train_window = None
        self.df = None
        self.features = None
        self.model_type = None
        self.model = None

    def prepare(
        self,
        df="clean",
        shopper: tuple = (0, 1999),
        week: tuple = (1, 90),
        product: tuple = (0, 249),
        price_aggregation_fn="mode",
        trend_windows: list = [1, 3, 5],
    ):
        """
        use:
            - preparing the data for the train test split

        preparation operations:
            - adjusting the data structure according to the final output
            - treating missing price values using replacement
            - feature creation:
                - weeks since last purchase
                - purchase trends - based on the trend window
                - product purchase frequency - based on all available data

        requirements:
            - 'prices' map needs to be stored at self.mappings['prices']
            - 'product_histories' map needs to be stored at
              self.mappings['product_histories']

        input:
            - df: pd.Dataframe or str
                - input dataframe on which the preparation operations should be performed
                - for str: input dataframe need to be stored at self.data[str]
                - default='clean': use the cleaned data frame stored at self.data['clean']
            - shopper: tuple or list
                - shopper ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a shopper id range
            - week: tuple or list
                - week ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a week id range
            - product: tuple or list
                - product ids which should be included in the prepared dataframe
                - for tuple: specified the first and last id in a product id range
            - price_aggregation_fn: function(list)
                - function used to aggregate the list of prices in the 'prices' map
                - default='mode': mode aggregation using scipy.stats.mode
            - trend_windows: list
                - list of trend window weeks for which a trend feature should be created

        return: pd.DataFrame
            - prepared data to be used for the train test split
        """
        import numpy as np
        import pandas as pd
        import itertools
        import time
        from tqdm import tqdm

        tqdm.pandas()

        start = time.time()
        if type(df) == str:
            df = self.data[df]

        # adjusting the data structure according to the final output
        # ------------------------------------------------------------------------------

        # converting tuples to ranges
        if type(shopper) != list:
            shopper = list(range(shopper[0], shopper[-1] + 1))
        self.shoppers = shopper
        if type(week) != list:
            week = range(week[0], week[-1] + 1)
        if type(product) != list:
            product = range(product[0], product[-1] + 1)

        # - creating the shopper, week, product combinations
        print(
            f"[prepare] itertools... (elapsed time: {self._format_time(time.time() - start)})"
        )
        output = pd.DataFrame(itertools.product(shopper, week, product))
        output.rename(columns={0: "shopper", 1: "week", 2: "product"}, inplace=True)

        print(
            f"[prepare] merge... (elapsed time: {self._format_time(time.time() - start)})"
        )
        if all([type(df) != pd.core.frame.DataFrame, df == None]):
            output["price"] = None
            output["discount"] = None
            output["purchased"] = None
        else:
            output = output.merge(
                df,
                how="left",
                left_on=list(output.columns),
                right_on=list(output.columns),
            )
            output["purchased"].fillna(0, inplace=True)
            output["discount"].fillna(0, inplace=True)

        # treating missing price values - replacement
        # ------------------------------------------------------------------------------
        print(
            f"[prepare] cleaning... (elapsed time: {self._format_time(time.time() - start)})"
        )
        price_map = self.aggregate_price_map(price_aggregation_fn, verbose=0)
        output.loc[output["price"].isna(), "price"] = output.loc[
            output["price"].isna(), :
        ].progress_apply(
            lambda row: price_map.loc[row["week"] - 1, str(int(row["product"]))], axis=1
        )

        # feature creation
        # ------------------------------------------------------------------------------
        print(
            f"[prepare] feature creation... (elapsed time: {self._format_time(time.time() - start)})"
        )

        # weeks since last purchase
        output["weeks_since_last_purchase"] = output.progress_apply(
            lambda row: self.get_last_purchase(
                int(row["shopper"]), str(int(row["product"])), row["week"]
            ),
            axis=1,
        )
        output["weeks_since_last_purchase"] = (
            output["week"] - output["weeks_since_last_purchase"]
        )
        output.loc[
            output["weeks_since_last_purchase"] == np.inf, "weeks_since_last_purchase"
        ] = np.ceil(output["week"].max() * 1.15)

        # purchase trend features
        for window in trend_windows:
            output["trend_" + str(window)] = output.progress_apply(
                lambda row: self.get_trend(
                    int(row["shopper"]), str(int(row["product"])), row["week"], window
                ),
                axis=1,
            )
        # product purchase frequency
        output["product_freq"] = output.progress_apply(
            lambda row: self.get_trend(
                int(row["shopper"]), str(int(row["product"])), row["week"], row["week"]
            ),
            axis=1,
        )

        # price elasticity
        elasticity_df = Elasticities()
        elasticity_df = elasticity_df.calc_week_elasticity(week)
        output = output.mereg(elasticity_df, how = 'left')
        
        
        # shopper coupon behaviour
        redemption_rates = Redemptions()
        output = redemption_rates.shopper_coupon_rates(output, week, shopper):

        # producrt discount behaviour
         
        

        
        
        # **************************************************************
        # feature creation: product cluster features, e.g. discount in complement/substitute
        # sascha
        # **************************************************************

        print(
            f"[prepare] done (elapsed time: {self._format_time(time.time() - start)})"
        )
        self.data["prepare"] = output
        return output

    def train_test_split(
        self,
        test_week: int,
        train_window: int,
        df="prepare",
        features: list = "default",
    ):
        """
        use:
            - further preparing the date to be used for model training and evaluation

        functionality:
            - splits the data into train and test data according to the specified test
              week and train window
            - WOE encoding for the shopper id and product id
            - separating target and features

        input:
            - train_week: int
                - test data: week for which the model should be tested
            - train_window: int
                - train data: number of week prior to the test week on which the model
                  will be trained
            - df: pd.DataFrame or str
                - input dataframe on which the split should be performed
                - default='prepare': used the data prepared by self.prepare() stored at
                  self.data['prepare']
            features: list
                - list of feature names which should be included in the model
                - default='default': drops redundant features, i.e. ['shopper', 'week',
                  'product', 'product_history', 'last_purchase']

        return: pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
            - X_train, y_train, X_test, y_test to be used for model training
        """
        import category_encoders as ce
        from IPython.display import clear_output

        self.test_week = test_week
        self.train_window = train_window
        self.df = df
        self.features = features

        if type(df) == str:
            df = self.data[df]

        # train and test split
        # ------------------------------------------------------------------------------
        start = test_week - train_window
        train = df.loc[(df["week"] >= start) & (df["week"] < test_week), :]
        test = df.loc[(df["week"] == test_week), :]

        # WOE category encoding
        # ------------------------------------------------------------------------------
        encoder = ce.WOEEncoder()
        train.loc[:, "shopper_WOE"] = encoder.fit_transform(
            train["shopper"].astype("category"), train["purchased"]
        )["shopper"].values
        test.loc[:, "shopper_WOE"] = encoder.transform(
            test["shopper"].astype("category")
        )["shopper"].values
        encoder = ce.WOEEncoder()
        train.loc[:, "product_WOE"] = encoder.fit_transform(
            train["product"].astype("category"), train["purchased"]
        )["product"].values
        test.loc[:, "product_WOE"] = encoder.transform(
            test["product"].astype("category")
        )["product"].values
        clear_output()

        # separating target and features
        # ------------------------------------------------------------------------------
        features = (
            [
                col
                for col in train.columns
                if col
                not in [
                    "purchased",
                    "shopper",
                    "week",
                    "product",
                    "product_history",
                    "last_purchase",
                ]
            ]
            if features == "default"
            else features
        )
        X_train = train.loc[:, features]
        y_train = train["purchased"]
        X_test = test.loc[:, features]
        y_test = test["purchased"]

        return X_train, y_train, X_test, y_test

    def fit(self, model_type: str, X_train, y_train, **kwargs):
        """
        use:
            - trains a model according to the specified model_type and train data

        input:
            - model_type: str
                - specifies which model should be used for model training
                - requirements: method will call a method called self._fit_#MODEL_TYPE#()
                  which needs to contain
                  all model training steps for the respective model_type and needs to
                  return the model object
            - X_train: pd.DataFrame
                - dataframe containing the train features
            - y_train: pd.DataFrame
                - dataframe containing the train target
            - **kwargs: keyword-argument dict
                - contains potential keyword arguments for the model training depending
                  on the model type

        return: object
            - trained model
        """
        self.model_type = model_type
        self.model = eval(f"self._fit_{model_type}(X_train, y_train, **kwargs)")
        return self.model

    def _fit_lgbm(self, X_train, y_train, **kwargs):
        """
        use:
            - trains a lightgbm classifier

        input:
            - X_train: pd.DataFrame
                - dataframe containing the train features
            - y_train: pd.DataFrame
                - dataframe containing the train target
            - **kwargs: keyword-argument dict
                - further meta parameter for the lgbm

        return: object
            - trained lgb-model object
        """
        import lightgbm

        model = lightgbm.LGBMClassifier()
        model.fit(X_train, y_train, **kwargs)
        return model

    def predict(self, model, X):
        """
        use:
            - predict a trained model based on the provided data

        input:
            - model: object
                - trained model
                - supported models: lightgbm classifier
            - X: pd.DataFrame
                - data for which the predictions should be made

        return: array-like
            - predicted purchase probabilities for the provided data
        """
        import lightgbm

        if type(model) == lightgbm.sklearn.LGBMClassifier:
            y_hat = model.predict_proba(X)[:, 1]
        return y_hat

    def score(self, y_true, y_hat, metric="log_loss"):
        """
        use:
            - calculating performance metrics for model evaluation

        input:
            - y_true: array-like
                - true value of the target
            - y_hat: array-like
                - predicted value of the target
            - metric: function(y_true, y_hat)
                - function that will calculate the metric/score
                - default='log-loss': calculates the log loss using sklearn.metrics.log_loss

        return:
            - model performance score
        """
        if metric == "log_loss":
            import sklearn

            metric = sklearn.metrics.log_loss

        return metric(y_true, y_hat)
