import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, \
    normalized_mutual_info_score
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class SegmentationEvaluator:

    @staticmethod
    def evaluate_silhouette(embeddings: np.array, cluster_labels: np.ndarray[int]) -> float:
        """
        Evaluates the quality of the clusters using Silhouette Score.
        :param embeddings: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: A float that represents the silhouette score.
        """
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        return silhouette_avg

    @staticmethod
    def evaluate_davies_bouldin(embeddings: np.array, cluster_labels: np.ndarray[int]):
        """
        Evaluates the quality of the clusters using Davies-Bouldin Index.
        :param embeddings: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: A float that represents the Davies-Bouldin Index.
        """
        # Check if embeddings need to be converted to a dense array
        if hasattr(embeddings, 'toarray'):
            embeddings = embeddings.toarray()

        dbi = davies_bouldin_score(embeddings, cluster_labels)
        return dbi

    @staticmethod
    def evaluate_calinski_harabasz(embeddings: np.array, cluster_labels: np.ndarray[int]):
        """
        Evaluates the quality of the clusters using the Calinski-Harabasz Index.
        :param embeddings: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: A float that represents the Calinski-Harabasz Index.
        """
        # Check if embeddings need to be converted to a dense array
        if hasattr(embeddings, 'toarray'):
            embeddings = embeddings.toarray()

        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
        return ch_score

    @staticmethod
    def evaluate_ari(true_labels: list, predicted_labels: list):
        """
        Evaluates the quality of the clusters using the Adjusted Rand Index (ARI).
        :param true_labels: The TF-IDF matrix representing the text data.
        :param predicted_labels: An array of cluster labels assigned to each document.
        :return: A float that represents the Adjusted Rand Index (ARI).
        """
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return ari

    @staticmethod
    def evaluate_nmi(true_labels: list, predicted_labels: list):
        """
        Evaluates the quality of the clusters using Normalized Mutual Information (NMI).
        :param true_labels: The TF-IDF matrix representing the text data.
        :param predicted_labels: An array of cluster labels assigned to each document.
        :return: A float that represents the Normalized Mutual Information (NMI).
        """
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        return nmi
