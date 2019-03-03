import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import networkx as nx

# Some Utilities
def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    minibatches = []
    X_shuff = np.copy(X)

    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]

def sample_negatives(X, n_e):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, whersample_negativese M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]

    corr = np.random.randint(n_e, size=M)
    e_idxs = np.random.choice([0, 2], size=M)

    X_corr = np.copy(X)
    X_corr[np.arange(M), e_idxs] = corr

    return X_corr

def accuracy(y_pred, y_true, thresh=0.5, reverse=False):
    """
    Compute accuracy score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.

    thresh: float, default: 0.5
        Classification threshold.

    reverse: bool, default: False
        If it is True, then classify (y <= thresh) to be 1.
    """
    y = (y_pred >= thresh) if not reverse else (y_pred <= thresh)
    return np.mean(y == y_true)

def auc(y_pred, y_true):
    """
    Compute area under ROC curve score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.
    """
    return roc_auc_score(y_true, y_pred)

# Taken from preprocessing.py file by Lucas Hu
# https://github.com/lucashu1/link-prediction/blob/master/gae/preprocessing.py
# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_isolates=True):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)

    if prevent_isolates == True:
        assert len(list(nx.isolates(g))) == 0 # no isolates in graph

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    all_edge_idx = [i for i in range(edges.shape[0])]
    np.random.shuffle(all_edge_idx)

    test_edges = []
    val_edges = []
    test_edge_idx = []
    val_edge_idx = []

    # Iterate over shuffled edges, add to train/val sets
    for edge_ind in all_edge_idx:
        edge = edges[edge_ind]
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would create an isolate, backtrack and move on
        if prevent_isolates == True:
            g.remove_edge(node1, node2)
            if len(nx.isolates(g)) > 0:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.append(edge)
            test_edge_idx.append(edge_ind)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.append(edge)
            val_edge_idx.append(edge_ind)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_isolates == True:
        assert len(list(nx.isolates(g))) == 0 # still no isolates in graph

    test_edges = np.array(test_edges)
    val_edges = np.array(val_edges)
    test_edge_idx = np.array(test_edge_idx)
    val_edge_idx = np.array(val_edge_idx)

    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Takes in numpy arrays
    # Returns whether or not anything in a is in b
    def ismember(a, b, tol=5):
        # a is empty --> return true automatically
        if type(a) is not np.ndarray and len(a) == 0:
            return True
        elif type(a) is np.ndarray and a.size == 0:
            return True

        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_j, idx_i], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if ismember([idx_j, idx_i], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])


    assert ~ismember(np.array(test_edges_false), edges_all)
    assert ~ismember(np.array(val_edges_false), edges_all)
    assert ~ismember(np.array(val_edges_false), np.array(train_edges_false))
    assert ~ismember(np.array(test_edges_false), np.array(train_edges_false))
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
	val_edges, val_edges_false, test_edges, test_edges_false



    def read_data(self):
        # Get Path of Data
        file = "/data/fazele/workplace/data/wordnet2/train2id.txt"
        path_train = os.getcwd() + file
        file = "/data/fazele/workplace/data/wordnet2/valid2id.txt"
        path_test = os.getcwd() + file
        file = "/data/fazele/workplace/data/wordnet2/test2id.txt"
        path_valid = os.getcwd() + file
        file = "/data/fazele/workplace/data/wordnet2/entity2id.txt"
        path_entity = os.getcwd() + file
        file = "/data/fazele/workplace/data/wordnet2/relation2id.txt"
        path_relation = os.getcwd() + file

        # Open Data Files
        train_file = open(path_train, "r")
        valid_file = open(path_valid, "r")
        test_file = open(path_test, "r")
        entity_file = open(path_entity, "r")
        relation_file = open(path_relation, "r")

        # Read Files of Data
        train_triple_file = train_file.read()
        train_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in train_triple_file.splitlines()]
        train_triple1 = np.array(train_triple_temp)
        self.n_train = int(train_triple1[0])
        train_triple1 = np.array(np.delete(train_triple1, 0))
        o = np.concatenate(train_triple1, axis=0)
        oo = np.reshape(o, (self.n_train, 3))
        train_triple = torch.from_numpy(oo).float().to(self.device)

        self.train_triple = train_triple

        # Save validation triples in vector
        valid_triple_file = valid_file.read()
        valid_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in valid_triple_file.splitlines()]
        valid_triple1 = np.array(valid_triple_temp)
        self.n_valid = int(valid_triple1[0])
        valid_triple1 = np.delete(valid_triple1, 0)

        ov = np.concatenate(valid_triple1, axis=0)
        oov = np.reshape(ov, (self.n_valid, 3))
        valid_triple = torch.from_numpy(oov).float().to(self.device)

        self.valid_triple = valid_triple

        # Save testing triples in vector
        test_triple_file = test_file.read()
        test_triple_temp = [np.array(s_inner.split(' '), dtype=int) for s_inner in test_triple_file.splitlines()]
        test_triple1 = np.array(test_triple_temp)
        self.n_test = int(test_triple1[0])
        test_triple1 = np.delete(test_triple1, 0)

        ot = np.concatenate(test_triple1, axis=0)
        oot = np.reshape(ot, (self.n_test, 3))
        test_triple = torch.from_numpy(oot).float().to(self.device)

        self.test_triple = test_triple

        # Save entity-index vector
        entity_tuple_file = entity_file.read()
        entity_tuple_temp = [np.array(s_inner.split('\t')) for s_inner in entity_tuple_file.splitlines()]
        entity_tuple = np.array(entity_tuple_temp)
        self.n_entity = int(entity_tuple[0])
        entity_tuple = np.array(np.delete(entity_tuple, 0))

        entity_tuple_final = np.empty([self.n_entity, 2], dtype='U20')
        for i in range(0, self.n_entity):
            entity_tuple_final[i] = entity_tuple[i]

        self.entity_id = entity_tuple_final

        # Save relation-index in vector
        relation_tuple_file = relation_file.read()
        relation_tuple_temp = [np.array(s_inner.split('\t')) for s_inner in relation_tuple_file.splitlines()]
        relation_tuple = np.array(relation_tuple_temp)
        self.n_relation = int(relation_tuple[0])
        relation_tuple = np.array(np.delete(relation_tuple, 0))

        relation_tuple_final = np.empty([self.n_relation, 2], dtype='U1000')
        for i in range(0, self.n_relation):
            relation_tuple_final[i] = relation_tuple[i]

        self.relation_id = relation_tuple_final
