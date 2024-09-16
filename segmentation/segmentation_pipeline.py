import os
import warnings
import argparse
import pandas as pd
from datetime import datetime

from segmentation import tfidf_kmeans, se3, sbert_spectral, llm
from utils import constants, logger_script, util_data_loader

warnings.filterwarnings("ignore")
logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class SegmentationPipeline:
    """
        A class dedicated to clustering legal document data from Rechtspraak case documents.

    This pipeline applies various data clustering techniques to legal case documents, utilizing different approaches
    such as TF-IDF with K-Means, self-segmentation, and spectral clustering to categorize headers, full texts,
    or specific sections based on the provided method.

    Attributes:
        tfidf_kmeans (TfidfKMeansClusterer): Clusters section headers and section content using TF-IDF and K-Means
                                            clustering. Includes methods for:
                                              - Clustering based on seed words.
                                              - Clustering based on labeled data.
        se3_segmenter (Se3Clusterer): Clusters full text content using a self-segmentation approach.
        transformer_spectral (TFSpectralClusterer): Clusters specific sections (tri-grams) of documents using
                                                    Transformer embeddings combined with Spectral Clustering.
        llm_clusterer (LLMClusterer): Clusters document sections using a Large Language Model.

    Methods:
        segmentation_process_selector(method: int, input_path: str):
            Selects and applies the appropriate clustering method based on the given method number.
            - method 1: Header Clustering using TF-IDF and K-Means with seed words.
            - method 2: Full Text Clustering using TF-IDF and K-Means with labeled data.
            - method 3: Section Clustering using Se3 self-segmentation.
            - method 4: Section Clustering using S-BERT and Spectral Clustering.
            - method 5: Section Clustering using LLM-based classification.
    """

    def __init__(self):
        """
        Initializes the SegmentationPipeline with the specific components for clustering and segmentation methods.
        """
        self.tfidf_kmeans = tfidf_kmeans.TfidfKMeansClusterer()
        self.se3_segmenter = se3.Se3Clusterer()
        self.transformer_spectral = sbert_spectral.TFSpectralClusterer()
        self.llm_clusterer = llm.LLMClusterer()

    def segmentation_process_selector(self, method: int, input_path: str, evaluate: bool, plot: bool):
        """
        Selects and executes a clustering method based on the provided method identifier. Depending on the method
        chosen, the function clusters different types of data, saves the results to a CSV file, and logs the outcome.
        :param method: An integer representing the clustering method to execute. The options are:
                1: Clusters headers using TF-IDF and K-Means with seed words and saves the results to a CSV file.
                2: Clusters full text using TF-IDF and K-Means with labeled data and saves the results to a CSV file.
                3: Clusters sections using Se3 self-segmentation and saves the results to a CSV file.
                4: Clusters sections using S-BERT and Spectral Clustering and saves the results to a CSV file.
                5: Clusters sections using LLM-based classification and saves the results to a CSV file.
        :param input_path: The path to the file to be processed.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :param plot: A bool that indicates whether a plot must be saved of clustered data.
        :return: The function performs data clustering and saves the results to a CSV file.
        """
        extracted_df = pd.DataFrame()
        method_name = ''

        # Load the input dataframe
        logger.info("Loading input data...")
        df_to_process = util_data_loader.load_csv_to_df(input_path)

        # Determine the processing method and execute the corresponding segmentation technique.
        logger.info("Start segmentation process...")
        match method:
            case 1:
                method_name = 'Tf-idf and K-means clusters using seed words'
                extracted_df = self.tfidf_kmeans.guided_kmeans_with_seed_words(df_to_process, evaluate, plot)
            case 2:
                method_name = 'Tf-idf and K-means clusters using labeled data'
                labeled_df = util_data_loader.load_csv_to_df(constants.LABELED_HEADERS_FILE_PATH)
                extracted_df = self.tfidf_kmeans.guided_kmeans_with_labeled(df_to_process, labeled_df, evaluate, plot)
            case 3:
                method_name = 'Se3 self-segmentation clusters'
                extracted_df = self.se3_segmenter.process_se3_segmentation(df_to_process, evaluate)
            case 4:
                method_name = 'S-BERT and Spectral Clustering clusters'
                extracted_df = self.transformer_spectral.process_sbert_spectral(df_to_process,
                                                                                constants.SEED_WORDS_LIST,
                                                                                evaluate)
            case 5:
                method_name = 'LLM classification'
                extracted_df = self.llm_clusterer.process_llm_segmentation(df_to_process)

        if extracted_df is not None and not extracted_df.empty:
            # Generate the current timestamp
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create the filename using the method number and the current time
            filename = f"{method}_{current_time}.csv"
            # Combine the folder path and the filename
            file_path = os.path.join(constants.SEGMENTATION_RESULTS_DIR, filename)
            # Save the DataFrame to the CSV file
            extracted_df.to_csv(file_path, index=False)
            logger.info(f"CSV with {method_name} saved to {file_path}!")
        else:
            error_message = "Segmentation failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Segmentation Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', type=int, choices=range(1, 6), default=1,
                        help=(
                            'Specify clustering method (1-5):\n'
                            '1 = TF-IDF + K-MEANS with seed words: clusters headers based on seed word groups,\n'
                            '2 = TF-IDF + K-MEANS with labeled data: clusters full text based on pre-labeled data,\n'
                            '3 = Self-Segmentation (Se3): clusters sections using the Se3 self-segmentation method,\n'
                            '4 = S-BERT + Spectral Clustering: clusters sections using S-BERT embeddings combined '
                            'with spectral clustering,\n'
                            '5 = LLM-based clustering: clusters sections using a Large Language Model-based approach.'
                        ))
    parser.add_argument('--input', type=str, default=constants.SECTIONS_PATH.format(year=2020),  # TODO: Make input dynamic
                        help="The path to the input data CSV file")
    parser.add_argument('--eval', action='store_true', help="If true, returns the evaluation scores in a CSV file.")
    parser.add_argument('--plot', action='store_true', help="If true, returns a plot of the clustered data (only "
                                                            "available for TFIDF + K-MEANS methods).")

    args = parser.parse_args()

    # Initialize the segmentation pipeline object and run the segmenting process
    logger.info("Start segmentation pipeline...")
    segmentation_pipe = SegmentationPipeline()
    segmentation_pipe.segmentation_process_selector(args.method, args.input, args.eval, args.plot)

    logger.info("Segmentation pipeline successfully finished!")
