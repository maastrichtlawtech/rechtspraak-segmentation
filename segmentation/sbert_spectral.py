import pandas as pd
import numpy as np
import segmentation_eval

from typing import List
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from utils import constants, logger_script, util_preprocessing
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class TFSpectralClusterer:
    """
    A class to perform text segmentation using Spectral Clustering with sentence embeddings obtained from a
    Transformer models. The clustering process segments the input documents based on their semantic similarity.

    Attributes:
        embedding_model (SentenceTransformer): The SentenceTransformer models used to compute sentence embeddings
                                               for the input documents.

    Methods:
        compute_embeddings(docs, batch_size=32, file_path=constants.EMBEDDINGS_SAVE_PATH, file_name="sent_embeddings",
                           force_compute=False):
            Computes or loads pre-computed sentence embeddings for the input documents using a Transformer models.

        process_sbert_spectral(input_df):
            Processes a DataFrame of input documents to compute sentence embeddings, perform spectral clustering, and
            assign cluster labels to the documents.
    """

    def __init__(self):
        """
        Initializes the TFSpectralClusterer with a SentenceTransformer models configured for computing embeddings.
        """
        self.embedding_model = SentenceTransformer(constants.DUTCH_BERT)

    @staticmethod
    def aggregate_embeddings(embeddings: List[np.ndarray], method: str = 'mean') -> np.ndarray:
        """
        Aggregates a list of embeddings into a single embedding.
        :param embeddings: A list of numpy arrays, each representing a trigram embedding.
        :param method: The method used for aggregation. Default is 'mean'. Other options include 'max'.
        :return: A numpy array representing the aggregated embedding.
        """
        if method == 'mean':
            # Compute the mean of the embeddings if 'mean' method is chosen
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            # Compute the maximum of the embeddings if 'max' method is chosen
            return np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def process_sbert_spectral(self,
                               input_df: pd.DataFrame,
                               seed_words_list: List[List[str]],
                               evaluate: bool) -> pd.DataFrame:
        """
        Processes input documents for segmentation using sentence embeddings and spectral clustering, with a bias
        towards certain seed words, working with trigrams.

        :param input_df: A pandas DataFrame containing the input documents to be segmented.
        :param seed_words_list: A list of seed word clusters to guide the clustering.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :return: A pandas DataFrame with clusters biased towards seed words.
        """
        # Create a subset of the DataFrame based on the proportions of 'instantie'
        subset_df = util_preprocessing.create_subset_based_on_proportions(input_df)

        # Apply tokenization to the full texts and generate tri-grams
        subset_df = util_preprocessing.tokenize_sentences(subset_df, constants.FULLTEXT_COL)
        trigrams = subset_df[constants.TOKENIZED_COL].apply(util_preprocessing.generate_trigrams)

        seed_embeddings = []
        for seed_words in seed_words_list:
            seed_embeddings.append(
                [self.embedding_model.encode(seed, normalize_embeddings=True) for seed in seed_words])

        aggregated_embeddings = []  # To store the aggregated trigram embeddings for each document
        trigram_sentences = []  # To store trigrams as lists of sentences for mapping back to clusters
        all_trigram_embeddings = []  # To store all individual trigram embeddings for bias calculation

        # Compute embeddings for each trigram and aggregate them
        for trigram_list in tqdm(trigrams, desc="Computing aggregated embeddings"):
            trigram_embeddings = [self.embedding_model.encode(' '.join(trigram), normalize_embeddings=True) for trigram
                                  in trigram_list]
            aggregated_embedding = self.aggregate_embeddings(trigram_embeddings, method='mean')

            aggregated_embeddings.append(aggregated_embedding)
            trigram_sentences.append(trigram_list)
            all_trigram_embeddings.extend(
                trigram_embeddings)  # Collect all trigram embeddings for similarity calculation

        # Convert the embeddings and seed embeddings to numpy arrays
        all_trigram_embeddings = np.array(all_trigram_embeddings)

        # Compute cosine similarity between trigram embeddings and seed word clusters
        seed_cluster_similarities = []
        for seed_cluster in seed_embeddings:
            cluster_embedding = np.mean(seed_cluster, axis=0)  # Average embedding of seed words in each cluster
            similarity_scores = cosine_similarity(all_trigram_embeddings, [cluster_embedding])
            seed_cluster_similarities.append(similarity_scores)

        # Concatenate similarity scores with original embeddings to form augmented embeddings
        augmented_embeddings = np.hstack([all_trigram_embeddings] + seed_cluster_similarities)

        # Apply Spectral Clustering with augmented embeddings
        num_clusters = min(max(2, len(trigrams) // 2), 8)
        cluster_model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
        labels = cluster_model.fit_predict(augmented_embeddings)

        if evaluate:
            # Evaluate the quality of the clustering
            logger.info("Evaluating the clusters...")
            silhouette = segmentation_eval.SegmentationEvaluator.evaluate_silhouette(augmented_embeddings, labels)
            db_index = segmentation_eval.SegmentationEvaluator.evaluate_davies_bouldin(augmented_embeddings, labels)
            ch_score = segmentation_eval.SegmentationEvaluator.evaluate_calinski_harabasz(augmented_embeddings, labels)
            logger.info(f"Silhouette Score: {silhouette:.4f}")
            logger.info(f"Davies-Bouldin Index: {db_index:.4f}")
            logger.info(f"Calinski-Harabasz Score: {ch_score:.4f}")

        # Create a dictionary that maps cluster labels to their sentences for each document
        cluster_to_sentences = []
        trigram_index = 0  # To track which trigram we are assigning during clustering

        for trigram_list in trigram_sentences:
            cluster_dict = {}
            for trigram in trigram_list:
                label = labels[trigram_index]  # Get the cluster label for the current trigram
                trigram_sentence = ' '.join(trigram)
                cluster_dict.setdefault(label, []).append(trigram_sentence)
                trigram_index += 1  # Move to the next trigram
            cluster_to_sentences.append(cluster_dict)

        # Add the cluster-to-sentence mapping as a new column to the DataFrame
        subset_df[constants.CLUSTER_COL] = cluster_to_sentences

        return subset_df
