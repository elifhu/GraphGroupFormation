import numpy as np
import networkx as nx
import pandas as pd 

def compute_eigenmap_pandas(df_marks, c_thresh=0.5, denom=5):
    corr = pd.DataFrame(df_marks.drop(columns=['No.']).to_numpy().astype(float).T).corr()

    # Compute the weight matrices
    W = corr.to_numpy()
    return laplacian_eigenmap(W, c_thresh, denom)

def get_dim3_eigenmap(spectral_matrix):
    return spectral_matrix[:, :3]

def load_eigenmaps_pandas(dfs, qfiles, c_thresh=0.5, denom=5):
    """load eigenmaps from file if they exist, otherwise compute them

    Args:
        dfs (list): list of dataframes
        qfiles (list): list of eigenmap files
        c_thresh (float, optional): threshold for correlation to exist as edge. Defaults to 0.5.
        denom (int, optional): denominator for weight calculation for graph edges. Defaults to 5.

    Returns:
        list: list of spectral matrices
    """
    qs = []
    for df, qfile in zip(dfs, qfiles):
        try:
            q = np.load(qfile)
        except FileNotFoundError:
            print('Eigenmaps not found. Computing...')
            q = get_dim3_eigenmap(compute_eigenmap_pandas(df, c_thresh, denom))
            np.save(qfile, q)
            print('Done.')
        qs.append(q)
    return qs

def compute_spectral_colors(q):
    colors = (q - q.min(axis=0)) / (q.max(axis=0) - q.min(axis=0))
    return colors

def laplacian_eigenmap(W, c_thresh=0.5, denom=5):
    """Compute eigenmap from a weighted adjacency matrix

    Args:
        W (np.array): weighted adjacency matrix

    Returns:
        np.array: spectral matrix
    """
    W[W < c_thresh] = 0
    W = np.exp(W**2/denom)
    W[W == 1.] = 0

    # Compute graph adjacencies
    A = W > 1

    # create the weighted graph using networkX
    G = nx.from_numpy_array(W)

    # get graph edges
    edges = G.edges()

    # get degree matrices
    degrees = [val for (node, val) in G.degree()]
    D = np.identity(len(degrees)) * degrees
    
    # compute normalised graph Laplacian using the weighted adjacency matrix
    L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G).toarray()

    # compute eigenvalues and eigenvectors, w and v, respectively
    w, v = np.linalg.eig(L)

    # the first eigenvalue is a maximally smooth constant, can be omitted
    spectral_matrix = v[:, 1:]
    return spectral_matrix