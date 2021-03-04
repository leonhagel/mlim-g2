import numpy as np
import pandas as pd

def reduce_mem_usage(name, df):
    """     
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    percentage = 100 * (start_mem - end_mem) / start_mem
    
    print(f'{percentage:.1f}% memory reduction for {name} (from {start_mem:.2f} MB to {end_mem:.2f} MB)') 

    return df


def parquet_loader(parquet_name, callback, path='../data/'):
    name = f'{parquet_name}.parquet.gzip'
    dataframe = None
    
    try:
        print(f'Read {name} from disk...')
        dataframe = pd.read_parquet(f'{path}{name}')
    except FileNotFoundError: 
        print(f'{name} was not found in {path}')
        print(f'Executing callback function "{callback.__name__}" to create {name}')
        dataframe = callback()
        dataframe.to_parquet(f'{path}{name}', compression='gzip')
        print(f'Wrote {name} to {path}')
    finally:
        print(f'Successfully loaded {name} into memory.')
    return dataframe
