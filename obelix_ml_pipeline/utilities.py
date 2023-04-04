# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def head(filename: str, n: int):
    try:
        with open(filename) as f:
            head_lines = [next(f).rstrip() for x in range(n)]
    except StopIteration:
        with open(filename) as f:
            head_lines = f.read().splitlines()
    return head_lines


def detect_delimiter(filename: str, n=3):
    # source: https://stackoverflow.com/questions/3952132/how-do-you-dynamically-identify-unknown-delimiters-in-a-data-file
    sample_lines = head(filename, n)
    common_delimiters = [',', ';', '\t', ' ', '|', ':']
    for d in common_delimiters:
        ref = sample_lines[0].count(d)
        if ref > 0:
            if all([ref == sample_lines[i].count(d) for i in range(1, n)]):
                return d
    return ','


def load_csv_or_excel_file_to_df(path_to_file):
    if path_to_file.endswith('.xlsx') or path_to_file.endswith('.XLSX'):
        df = pd.read_excel(path_to_file)
    elif path_to_file.endswith('.csv') or path_to_file.endswith('.CSV'):
        df = pd.read_csv(path_to_file, sep=detect_delimiter(path_to_file))
    else:
        raise ValueError(f'{path_to_file} is not a valid csv or excel file')

    return df


def merge_dfs(df1, left_index, df2, right_index):
    return df1.merge(df2, left_on=[left_index], right_on=[right_index])


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_dendrogram_for_substrate_rep(df, representation_type):
    # for plotting drop all columns containing smiles
    df = df.loc[:, ~df.columns.str.contains('SMILES|smiles')]
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, 1:].values)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X)
    plt.figure()
    plt.title(f"Hierarchical Clustering Dendrogram {representation_type}")
    # plot the top n levels of the dendrogram
    try:
        plot_dendrogram(model, labels=df['index'].values, truncate_mode="level", p=5)
    except KeyError:
        # in this case the experimental df is present and the Substrate column is used as label
        plot_dendrogram(model, labels=df['Substrate'].values, truncate_mode="level", p=5)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()