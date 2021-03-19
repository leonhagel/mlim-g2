import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.cluster import AgglomerativeClustering


class CooccurencePrinter:
    def __init__(self, data, group_var="basket", member_var="product"):
        self.data = data
        self.group_var = group_var
        self.member_var = member_var
        self.member_1 = member_var + '_1'
        self.member_2 = member_var + '_2'
        self.value = 'co-occurrence'
        self.table = self.compute_table()

    def co_occurrences_sparse(self):
        x = self.data
        group = self.group_var
        member = self.member_var
        
        row = x[group].values
        col = x[member].values
        dim = (x[group].max()+1, x[member].max()+1)

        group_member_table = scipy.sparse.csr_matrix(
            (np.ones(len(row), dtype=int), (row, col)),
            shape=dim
        )
        co_occurrences_sparse = group_member_table.T.dot(group_member_table).tocoo()
        co_occurrences_df = pd.DataFrame({
            self.member_1: co_occurrences_sparse.row,
            self.member_2: co_occurrences_sparse.col,
            self.value: co_occurrences_sparse.data,
        })
        return co_occurrences_df
        
    def compute_table(self):
        co_occurrences_df = self.co_occurrences_sparse()
        pivot_table = co_occurrences_df.pivot(
            index   = self.member_1, 
            columns = self.member_2, 
            values  = self.value
        ).fillna(0)
        return pivot_table

    def plot_heatmap(self, figsize=(10,10)):
        '''
        we clip the large self-co-occurences (diagonal values)
        to the max non-self coocuerence (non-diagonal value)
        '''
        table = self.table
        np.fill_diagonal(table.values, 0)
        max_non_diagonal = table.mean().mean()
        np.fill_diagonal(table.values, max_non_diagonal)
        dist_table = 1 - table
        matfig = plt.figure(figsize=figsize)
        plt.matshow(dist_table, cmap=plt.cm.Blues, fignum=matfig.number)
        title = f"{self.member_var}-{self.value} heatmap"
        plt.title(title)
        plt.xlabel(self.member_1)
        plt.ylabel(self.member_2)
        plt.show() 

    def agglomerative_clustering(self, table):
        table_np = table.to_numpy()
        # ward minimizes the variance of the clusters being merged
        model = AgglomerativeClustering(linkage='ward', n_clusters=3).fit(table_np)
        new_order = np.argsort(model.labels_)
        ordered_table = table_np[new_order]
        ordered_table = ordered_table[:,new_order]
        return ordered_table

    def plot_ordered_heatmap(self):
        table = self.table
        ordered_table = self.agglomerative_clustering(table)
        matfig = plt.figure(figsize=(10,10))
        plt.matshow(ordered_table, cmap=plt.cm.Blues, fignum=matfig.number)
        plt.title("Ordered Co-Clustering of Category Purchases")
        plt.xlabel("product_cat_1")
        plt.ylabel("product_cat_2")