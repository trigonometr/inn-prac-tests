import hnswlib
import numpy as np

from typing import Optional


class Matcher:
    """
    Matches given vectors in data with nearest vectors in dataset
    provided by path_to_index.

    Parameters
    ----------
    dim : dimensionality of vectors

    max_elements : max vectors amount for the matching session

    M : parameter that defines the maximum number of outgoing
        connections in the graph

    ef_construction : parameter that controls speed/accuracy trade-off
        during the index construction

    path_to_index : path to saved file index

    """

    def __init__(
        self,
        dim: int,
        max_elements: int,
        M: Optional[int] = None,
        ef_construction: Optional[int] = None,
        path_to_index: Optional[str] = None,
    ):
        self.dim = dim
        self.path_to_index = path_to_index
        self.max_elements = max_elements
        self.index = hnswlib.Index(space="l2", dim=self.dim)

        if self.path_to_index:
            self.index.load_index(
                path_to_index=self.path_to_index,
                max_elements=self.max_elements,
            )
        else:
            self.index.init_index(
                max_elements=self.max_elements,
                M=M,
                ef_construction=ef_construction,
            )

    def add_items(self, data: np.ndarray):
        """
        Adds data to index
        """

        self.index.add_items(data)

    def save_index(self, path_to_index: str):
        """
        Saves self.index to path_to_index
        """

        self.index.save_index(path_to_index)
        self.path_to_index = path_to_index

    def get_nearest_neighbour(self, data: np.ndarray) -> np.ndarray:
        """
        Finds nearest neighbours in indexed-dataset from self.index
        for every data's row

        Parameters
        ----------
        data : len(data.shape) == 2 and data.shape[1] == self.dim
            the first axis states the amounts of vectors

        Returns
        -------
        labels : labels of the matched (nearest) vectors
            for data.shape[0] vectors
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Expected data to be np.ndarray," f"instead got: {type(data)}"
            )

        if len(data.shape) != 2 or data.shape[1] != self.dim:
            raise ValueError(
                f"Wrong shape must be 2-dim with {self.dim} on the second axis"
            )

        labels, distances = self.index.knn_query(data, k=1)
        return labels
