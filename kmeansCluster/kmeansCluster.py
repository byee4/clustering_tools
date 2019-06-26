#!/usr/bin/env python

"""
basic module to use as a boilerplate for clustering methods that require
building an adjacency graph first.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans_clustering(fn, out, k, seed):
    df = read_exprs_as_df(fn).T
    arr = df.values
    km = KMeans(n_clusters=k)
    km.fit(arr)

    labels = km.labels_
    results = pd.DataFrame([df.index, labels]).T
    results.columns = ['Barcode','Cluster']
    results.to_csv(out, sep=',', index=False)
    exit(0)

def read_exprs_as_df(fn):
    """
    Reads in a file as a dataframe, expects genes as rownames, cells as colnames

    :param fn:
    :return:
    """
    df = pd.read_table(fn, index_col=0)
    return df

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
    parser.add_argument(
        "--outfile",
        required=True,
        help="output filename (tabbed cells and clusterID)"
    )
    parser.add_argument(
        "--k",
        required=True,
        help="k",
        type=int,
        default=3
    )
    parser.add_argument(
        "--seed",
        required=False,
        help="random seed for starting centroid (default 1)",
        default=1,
    )
    args = parser.parse_args()

    in_file = args.infile
    out_file = args.outfile
    k = args.k
    seed = args.seed


    run_kmeans_clustering(in_file, out_file, k, seed)

if __name__ == "__main__":
    main()
