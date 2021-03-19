import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


class P2VDataLoader:
    
    def __init__(
        self,
        baskets: pd.DataFrame,
        basket_var = 'basket',
        product_var = 'product'
    ):
        self.basket_var = basket_var
        self.product_var = product_var
        self.load(baskets)
    
    
    def load(self, baskets):
        # this takes 15min
        self.output = self.baskets_to_list(baskets)

    
    def baskets_to_list(self, baskets):
        baskets['basket'] = baskets.groupby(['week', 'shopper']).ngroup()
        baskets_grouped = baskets.groupby(self.basket_var)[self.product_var]
        baskets_list = baskets_grouped.agg(self.basket_to_list).tolist()
        return baskets_list
        

    def basket_to_list(self, basket):
        return basket.astype(str).tolist()


    
class P2VModel:
    def __init__(self, baskets, embedding_dimensions):
        w2v_config = {
            'sentences': baskets,
            'min_count': 2,
            'window': len(max(baskets, key=len)),
            'size': embedding_dimensions,
            'workers': 3,
            'sg': 1
        }
        self.fit(w2v_config)


    def fit(self, w2v_config):
        try:
            self.model = Word2Vec.load("../cache/p2v.model")
        except IOError:
            self.model = Word2Vec(**w2v_config)
            self.model.save("../cache/p2v.model")
            
            
    def get_embeddings(self):
        return self.model.wv[self.model.wv.vocab]
             
        
    def most_similar_products(self, product):
        return self.model.wv.most_similar(product)
        
        
    def tsne_plot(self, n_components=2, perplexity=4, random_state=0, figsize=(15,15)):
        result = TSNE(
            n_components = n_components, 
            perplexity = perplexity, 
            random_state = random_state
        ).fit_transform(self.get_embeddings())
        plt.figure(figsize=figsize)
        plt.scatter(result[:, 0], result[:, 1])
        products = list(self.model.wv.vocab)
        for i, product in enumerate(products):
            plt.annotate(product, xy=(result[i, 0], result[i, 1]))
        plt.show()