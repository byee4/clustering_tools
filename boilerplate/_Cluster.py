#!/usr/bin/env python

"""
basic module to use as a boilerplate for clustering methods that require
building an adjacency graph first.
"""
import argparse
import CLUSTERING MODULE
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pyflannCluster

def run_generic_clustering(fn, algorithm):
    df = read_exprs_as_df(fn)
    k = get_k(df)
    graph = get_sparse_knn_graph(df, k, algorithm)
    """
    CLUSTERING LOGIC GOES HERE
    """
    print("success")
    exit(0)

def get_sparse_knn_graph(df, k, algorithm):
    X = np.array(df)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm)
    knn_graph = nbrs.kneighbors_graph(X)
    return knn_graph

def read_exprs_as_df(fn):
    """
    Reads in a file as a dataframe, expects genes as rownames, cells as colnames

    :param fn:
    :return:
    """
    df = pd.read_table(fn)
    return df

def get_k(df):
    """
    Returns a value K which scales logarithmically to the number of cells
    in a sample.

    :param df: pandas.DataFrame
        table of rows=genes, cols=barcode/cells
    :return k: int
        number corresponding to log of cellcount.
    """

    return int(np.log(len(df.columns)))

def main():
    """
    Main program.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        required=True,
        help="input filename (tabbed file rows=genes, cols=cells) for now..."
    )

    args = parser.parse_args()

    in_file = args.infile
    algorithm = 'ball_tree'
    run_generic_clustering(in_file, algorithm)

if __name__ == "__main__":
    main()
