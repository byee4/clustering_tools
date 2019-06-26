#!/usr/bin/env python

"""
Implementation of the louvain partition method for clustering gene expression
matrices.
"""
import argparse
import louvain
import igraph
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def run_louvain(fn, algorithm):
    """
    Runs the main function, including running the algorithm

    :param fn:
    :param algorithm:
    :return:
    """
    df = read_exprs_as_df(fn)
    k = get_k(df)
    print("building graph...")
    indices, adj_matrix = get_sparse_knn_graph(df, k, algorithm)
    igraph = convert_sparse_to_igraph(indices, adj_matrix)
    print("running louvainCluster find partition...")
    part = louvain.find_partition(igraph, method='Modularity')
    exit(0)


def convert_sparse_to_igraph(indices, matrix):
    """
    Convert an adjacency graph in scipy sparse matrix
    format into an iGraph format.

    https://groups.google.com/forum/#!topic/network-analysis-with-igraph/gXHzgrtdxvE
    https://stackoverflow.com/questions/29655111/igraph-graph-from-numpy-or-pandas-adjacency-matrix

    :param matrix: numpy.array
    :return graph: igraph.Graph
        iGraph representation of a weighted sparse adj matrix
    """
    # sources, targets = matrix.nonzero()
    # weights = matrix[sources, targets]
    # weights = np.array(weights)[0]
    # print(dir(louvain))
    # ig = igraph.Graph(zip(sources, targets), directed=True,
    #                edge_attrs={'weight': weights})
    # return ig
    g = igraph.Graph.Adjacency((matrix > 0).tolist())
    g.es['weight'] = matrix[matrix.nonzero()]
    # g.vs['label'] = node_names  # or a.index/a.columns
    return g


def get_sparse_knn_graph(df, k, algorithm):
    """
    Given a gene expression matrix, find the k nearest neighbors

    :param df: pandas.DataFrame
        expression matrix (genes as rows, barcode/cells as columns)
    :param k: int
        k value for knn
    :param algorithm: string
        ie. 'ball_tree'
    :return:
    """
    print("df shape", df.shape)
    X = np.array(df)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(X)
    distances, indices = nbrs.kneighbors(X)
    knn_graph = nbrs.kneighbors_graph(X).toarray()
    print("indices, graph", len(indices), knn_graph.shape)
    return indices, knn_graph


def read_exprs_as_df(fn):
    """
    Reads in a file as a dataframe, expects genes as rownames, cells as colnames

    :param fn: string
    :return df: pandas.DataFrame
    """
    df = pd.read_table(fn, index_col=0).T
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
    run_louvain(in_file, algorithm)

if __name__ == "__main__":
    main()
