import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_eval

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import constants, logger_script

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class TfidfKMeansClusterer:
    """
    A class that combines TF-IDF vectorization with K-Means clustering to segment legal document headers.

    This class implements a semi-supervised approach to cluster headers from legal documents into predefined categories
    using TF-IDF features and K-Means clustering. The approach leverages seed embeddings based on legal topic keywords
    to guide the clustering process towards meaningful groupings.

    Attributes:
        vectorizer (TfidfVectorizer): A TF-IDF vectorizer used to transform text data into numerical vectors.
        stop_words (List[str]): A list of stopwords used by the vectorizer to ignore common words.
    """

    def __init__(self):
        """
        Initializes the TfidfKMeansClusterer with Dutch stopwords and additional custom stopwords.
        Sets up the TF-IDF vectorizer with these stopwords.
        """
        self.stop_words = stopwords.words('dutch') + constants.ADDITIONAL_DUTCH_STOPWORDS
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)

    @staticmethod
    def extract_headers_for_tfidf(df: pd.DataFrame) -> list:
        """
        Extracts 'header' values from the 'sections' nested dictionaries within the DataFrame.
        :param df: Input DataFrame containing a 'sections' column with nested dictionaries.
        :return: A list of extracted headers in lowercase.
        """
        header_values = []
        for nested_dict in df['sections']:
            for key, value in nested_dict.items():
                if key.isdigit():  # Check if the key is a number
                    header_values.append(value['header'].lower())
        return header_values

    @staticmethod
    def update_sections_with_labels(input_df: pd.DataFrame, cluster_labels: np.ndarray[int]) -> pd.DataFrame:
        """
        Updates the 'sections' column in the input DataFrame by adding a 'label' key with the cluster number.
        :param input_df: The original input DataFrame containing the 'sections' column.
        :param cluster_labels: The cluster labels to be added to the 'sections' column.
        :return: The updated DataFrame with the 'label' key added to each nested dictionary in the 'sections' column.
        """
        for idx, nested_dict in enumerate(input_df['sections']):
            for key in nested_dict:
                if key.isdigit():  # Check if the key is a number
                    # Add the cluster label to the dictionary
                    nested_dict[key]['label'] = int(cluster_labels[idx])
        return input_df

    @staticmethod
    def apply_kmeans(init_value: np.ndarray[int], tfidf_matrix: np.ndarray[int]) -> np.ndarray[int]:
        """
        Applies K-Means clustering to the TF-IDF matrix using the provided seed matrix for initialization.
        :param init_value: The seed matrix used to initialize the K-Means centroids.
        :param tfidf_matrix: The TF-IDF matrix representing the text data.
        :return: An array of cluster labels assigned to each document in the TF-IDF matrix.
        """
        kmeans = KMeans(n_clusters=len(constants.SEED_WORDS_LIST), init=init_value, n_init=1, random_state=42)
        kmeans.fit(tfidf_matrix)
        cluster_labels = kmeans.predict(tfidf_matrix)
        return cluster_labels

    @staticmethod
    def generate_tfidf_kmeans_scatter_plot(tfidf_matrix: np.array, cluster_labels: np.ndarray[int]):
        """
        Generates a 2D scatter plot of the TF-IDF matrix data points, colored by their cluster labels.
        :param tfidf_matrix: The TF-IDF matrix representing the text data.
        :param cluster_labels: An array of cluster labels assigned to each document.
        :return: None. Shows the plot and saves it as PNG file.
        """
        # Plotting the new data points and centroids after clustering
        logger.info("Plotting data...")

        # Reduce dimensions to 2D using PCA for visualization
        pca = PCA(n_components=2)
        new_data_2d = pca.fit_transform(tfidf_matrix.toarray())

        # Configure the plot settings and show it
        plt.figure(figsize=(10, 6))
        plt.scatter(new_data_2d[:, 0], new_data_2d[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                    label='New Data')
        plt.title("New Data Clustering Based on Refined TF-IDF Vectors")
        plt.legend()
        plt.show()  # TODO: add save statement

    def guided_kmeans_with_seed_words(self, input_df: pd.DataFrame, evaluate: bool, plot: bool) -> pd.DataFrame:
        """
        Applies K-Means clustering to the headers in the input DataFrame using seed words to guide the clustering.
        :param input_df: The input DataFrame containing the 'sections' column with nested dictionaries.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :param plot: A bool that indicates whether a plot must be saved of clustered data.
        :return: A DataFrame with headers and their corresponding cluster labels.
        """
        # Extract 'header' values from the nested dictionaries
        logger.info("Extracting the headers from input dataframe...")
        header_values = self.extract_headers_for_tfidf(input_df)

        # Fit the vectorizer to the headers and transform the headers
        logger.info("Fitting the Tf-idf vectorizer...")
        tfidf_matrix = self.vectorizer.fit_transform(header_values)

        # Vectorize the seed words
        logger.info("Vectorizing the seed words...")
        seed_vectors = []
        for seed_words in constants.SEED_WORDS_LIST:
            seed_vector = self.vectorizer.transform(seed_words).mean(axis=0)
            seed_vectors.append(seed_vector)

        seed_matrix = np.array(seed_vectors)
        seed_matrix = np.squeeze(seed_matrix)

        # Apply K-Means clustering
        logger.info("Starting K-Means clustering process...")
        cluster_labels = self.apply_kmeans(seed_matrix, tfidf_matrix)
        logger.info("K-Means clustering finished!")

        # Update the sections with cluster labels
        logger.info("Updating sections with cluster labels...")
        result_df = self.update_sections_with_labels(input_df, cluster_labels)

        if evaluate:
            # Evaluate the quality of the clustering
            logger.info("Evaluating the clusters...")

            silhouette = segmentation_eval.SegmentationEvaluator.evaluate_silhouette(tfidf_matrix, cluster_labels)
            db_index = segmentation_eval.SegmentationEvaluator.evaluate_davies_bouldin(tfidf_matrix, cluster_labels)
            ch_score = segmentation_eval.SegmentationEvaluator.evaluate_calinski_harabasz(tfidf_matrix, cluster_labels)

            logger.info(f"Silhouette Score: {silhouette:.4f}")
            logger.info(f"Davies-Bouldin Index: {db_index:.4f}")
            logger.info(f'Calinski-Harabasz Score: {ch_score:.4f}')

        if plot:
            logger.info("Generating the cluster plot...")
            self.generate_tfidf_kmeans_scatter_plot(tfidf_matrix, cluster_labels)

        return result_df

    def guided_kmeans_with_labeled(self,
                                   input_df: pd.DataFrame,
                                   labeled_df: pd.DataFrame,
                                   evaluate: bool,
                                   plot: bool) -> pd.DataFrame:
        """
        Applies K-Means clustering to the headers in the input DataFrame using refined TF-IDF vectors
        derived from labeled data.
        :param input_df: The input DataFrame containing the 'sections' column with nested dictionaries.
        :param labeled_df: The labeled DataFrame containing 'Header' and 'Cluster' columns.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :param plot: A bool that indicates whether a plot must be saved of clustered data.
        :return: A DataFrame with headers and their predicted cluster labels.
        """
        # Extract 'header' and 'cluster' values from the labeled DataFrame
        logger.info("Extract labeled data from CSV file...")
        labeled_headers = labeled_df['Header'].str.lower().tolist()
        labeled_clusters = labeled_df['Cluster'].tolist()

        # Fit the vectorizer on the entire labeled corpus
        logger.info("Fit Tf-idf vectorizer on labeled headers")
        self.vectorizer.fit(labeled_headers)

        # Initialize empty dictionary to store refined TF-IDF vectors for each cluster
        refined_tfidf_vectors = {}
        unique_clusters = sorted(set(labeled_clusters))

        # Process each cluster separately
        logger.info("Generate clusters from labeled data...")
        for cluster in tqdm(unique_clusters, desc="Processing clusters"):
            # Get headers for this cluster
            cluster_indices = [i for i, label in enumerate(labeled_clusters) if label == cluster]
            cluster_headers = [labeled_headers[i] for i in cluster_indices]

            # Get headers for the remaining data (i.e., all other clusters)
            remaining_indices = [i for i, label in enumerate(labeled_clusters) if label != cluster]
            remaining_headers = [labeled_headers[i] for i in remaining_indices]

            # Compute TF-IDF vectors for this cluster and remaining data using the unified vocabulary
            cluster_tfidf_matrix = self.vectorizer.transform(cluster_headers)
            remaining_tfidf_matrix = self.vectorizer.transform(remaining_headers)

            # Subtract the remaining TF-IDF vectors from the cluster-specific TF-IDF vectors
            refined_tfidf_matrix = cluster_tfidf_matrix.mean(axis=0) - remaining_tfidf_matrix.mean(axis=0)
            refined_tfidf_vectors[cluster] = refined_tfidf_matrix

        # Extract 'header' values from the nested dictionaries
        logger.info("Extracting the headers from input dataframe...")
        header_values = self.extract_headers_for_tfidf(input_df)

        # Compute the TF-IDF vectors for the extracted headers
        logger.info("Compute Tf-idf vectors for extracted headers...")
        tfidf_matrix = self.vectorizer.transform(header_values)

        distance_matrix = np.zeros((len(header_values), len(unique_clusters)))

        # Create a mapping from cluster labels to matrix indices
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

        # Calculate the distance from each header to each cluster's refined vector
        logger.info("Calculating distances between headers and clusters and assigning headers to clusters...")
        for cluster, refined_vector in refined_tfidf_vectors.items():
            i = cluster_to_index[cluster]  # Get the correct index for the current cluster
            refined_vector_dense = np.squeeze(np.asarray(refined_vector))
            distance_matrix[:, i] = tfidf_matrix.dot(refined_vector_dense.T)

        # Log the completion
        logger.info("Distance calculation complete. Assigned headers to clusters.")

        # Assign each header to the closest cluster based on the distance matrix
        cluster_labels = np.argmax(distance_matrix, axis=1)

        # Update the sections with cluster labels
        logger.info("Updating sections with cluster labels...")
        result_df = self.update_sections_with_labels(input_df, cluster_labels)

        if evaluate:
            # Evaluate the quality of the clustering
            logger.info("Evaluating the clusters...")

            # Find all occurrences of each labeled header in header_values
            labeled_true_clusters = []
            labeled_predicted_clusters = []

            for header, true_cluster in zip(labeled_headers, labeled_clusters):
                # Find all occurrences of this labeled header in the input headers
                occurrences = [i for i, h in enumerate(header_values) if h.lower() == header]
                # Append the true cluster and predicted cluster for each occurrence
                labeled_true_clusters.extend([true_cluster] * len(occurrences))
                labeled_predicted_clusters.extend([cluster_labels[i] for i in occurrences])

            # Evaluate internal metrics (on full data)
            silhouette = segmentation_eval.SegmentationEvaluator.evaluate_silhouette(tfidf_matrix, cluster_labels)
            db_index = segmentation_eval.SegmentationEvaluator.evaluate_davies_bouldin(tfidf_matrix, cluster_labels)
            ch_score = segmentation_eval.SegmentationEvaluator.evaluate_calinski_harabasz(tfidf_matrix, cluster_labels)

            # Evaluate external metrics (on duplicated labeled data)
            ari = segmentation_eval.SegmentationEvaluator.evaluate_ari(labeled_true_clusters,
                                                                       labeled_predicted_clusters)
            nmi = segmentation_eval.SegmentationEvaluator.evaluate_nmi(labeled_true_clusters,
                                                                       labeled_predicted_clusters)

            # Log evaluation results
            logger.info(f'Calinski-Harabasz Score: {ch_score:.4f}')
            logger.info(f"Silhouette Score: {silhouette:.4f}")
            logger.info(f"Davies-Bouldin Index: {db_index:.4f}")
            logger.info(f'Adjusted Rand Index: {ari:.4f}')
            logger.info(f'Normalized Mutual Information: {nmi:.4f}')

        if plot:
            logger.info("Generating the cluster plot...")
            self.generate_tfidf_kmeans_scatter_plot(tfidf_matrix, cluster_labels)

        return result_df
