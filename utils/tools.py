import numpy as np
import cupy as cp
from cupyx.scipy.spatial import distance
from datetime import datetime
def best_uint_type(x):
    # Check the number of bits needed to represent x as an unsigned int
    if x < 2**16:
        return np.uint16
    elif x < 2**32:
        return np.uint32
    else:
        return np.uint64

def compute_distances_cp_chunked(
        words_embeddings,
        vocabulary,
        metric: str = "euclidean",
        chunk_size: int = -1, # How many words embeddings per computation chunk
        dtype: np.dtype = np.float16 # Precision of the distances
    ) -> np.ndarray:
    """Computes the distances between each words embedding and the entire vocabulary.
    Benefit from cupy for a faster computation on GPU. Computes chunk-by-chunk to avoid
    overloading the VRAM."""
    number_of_words = words_embeddings.shape[0]

    if chunk_size == -1:
        # Calculate the ideal chunk_size such that storing the distances before they are exported
        # to RAM uses at most 2GB of VRAM. Note that cupyx.scipy.spatial.distance.cdist(x1, x2) 
        # first allocates an array of shape (x1.shape[0], x2.shape[0]) with np.float64 precision (8 bytes). We subsequently cast it to dtype.
        chunk_size = max(1, round(2e9/(vocabulary.shape[0]*8)))

    # Copy the vocabulary to GPU. Returns a reference to the original array if
    # it is already on GPU. 
    vocab_cp = cp.asarray(vocabulary)

    # Distances will be stored on RAM as a numpy array
    distances = np.empty((words_embeddings.shape[0], vocab_cp.shape[0]), dtype=dtype)

    for i in range(0, number_of_words, chunk_size):
        j = min(i+chunk_size, number_of_words)
        distances[i:j,:] = distance.cdist(words_embeddings[i:j], vocab_cp, metric).astype(dtype).get()
    
    return distances

def compute_distances(
        words_embeddings,
        vocabulary,
        metric: str = "euclidean",
        dtype: np.dtype = np.float16 # Precision of the distances
    ) -> np.ndarray:
    return compute_distances_cp_chunked(words_embeddings, vocabulary, metric, dtype=dtype)

def rank_neighbors(
        embeddings,
        vocabulary,
        distance_metric: str = "euclidean",
    ) -> np.ndarray:
    """For each embedding, ranks the elements in the vocabulary according to their distance
    with the embedding. Returns a numpy array of shape (embeddings.shape[0], vocabulary.shape[0]) 
    where array[i][j] contains the rank of the j-th vocabulary element in the list of neighbors of 
    the i-th embedding. The function benefits from cupy for a faster computation of distances and 
    sorting on GPU. Computes chunk-by-chunk to avoid overloading the VRAM."""

    # Copy the vocabulary to GPU. Returns a reference to the original array if
    # it is already on GPU. 
    vocab_cp = cp.asarray(vocabulary)

    # Ranks will be stored on RAM as a numpy array
    words_neighbors_ranked = np.empty((embeddings.shape[0], vocab_cp.shape[0]), dtype=best_uint_type(vocab_cp.shape[0]))
    i_max = embeddings.shape[0]

    # Calculate the ideal chunk_size such that storing the distances and the results of argsort before it 
    # is exported to RAM uses at most 2GB of VRAM. Note that cupyx.scipy.spatial.distance.cdist(x1, x2) 
    # first allocates an array of shape (x1.shape[0], x2.shape[0]) with np.float64 precision (8 bytes).
    # Argsort returns an array of int64 (8 bytes).
    chunk_size = max(1, round(2e9/(vocab_cp.shape[0]*8*2)))
    
    for i in range(0, i_max, chunk_size):
        j = min(i+chunk_size, i_max)
        words_neighbors_ranked[i:j,:] = distance.cdist(embeddings[i:j], vocab_cp, distance_metric).argsort(axis=-1).argsort(axis=-1).get() #argsort on GPU before copying to RAM with .get()
    
    return words_neighbors_ranked

def argsort_chunked(
        array: np.ndarray,
        dtype: np.dtype,
        chunk_size: int = 500
    ) -> np.ndarray:
    """Performs an argsort(axis=-1) on the array, chunk by chunk. Only useful
    if the dtype of the result is smaller than int64. In this case,
    rather than overloading the RAM with an int64 array -which will
    then require a copy when .astype() is called-, this function does it chunk
    by chunk."""

    argsorted_array = np.empty((array.shape), dtype=dtype)

    for i in range(0, array.shape[0], chunk_size):
        j = min(i+chunk_size, array.shape[0])
        argsorted_array[i:j] = array[i:j].argsort(axis=-1)

    return argsorted_array