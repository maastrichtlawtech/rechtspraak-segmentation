import os
import argparse
import pandas as pd
from datetime import datetime

import extractive_summarization, abstractive_summarization
from utils import constants, logger_script, util_data_loader, util_preprocessing

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


class SummarizationPipeline:
    """
        A class dedicated to summarizing legal document data from Rechtspraak case documents.

    This pipeline applies various summarization techniques to legal case documents, utilizing different approaches
    such as TextRank, BERT-based extractive summarization, and BART-based abstractive summarization.

    Attributes:
        extractive_summarizer (ExtractiveSummarizer): Applies extractive summarization methods like TextRank and BERT.
        abstractive_summarizer (AbstractiveSummarizer): Applies abstractive summarization methods like BART and Llama.

    Methods:
        summarization_process_selector(method: int, input_path: str, evaluate: bool):
            Selects and applies the appropriate summarization method based on the given method number.
            - method 1: Apply TextRank for extractive summarization.
            - method 2: Apply BERT for extractive summarization.
            - method 3: Apply BART for abstractive summarization.
            - method 4: Apply Llama for abstractive summarization.
    """

    def __init__(self):
        """
        Initializes the SummarizationPipeline with specific components for extractive and abstractive summarization
        methods.
        """
        self.textrank_summarizer = extractive_summarization.TextRankSummarizer()
        self.lsa_summarizer = extractive_summarization.LSAExtractiveSummarizer()
        self.abstractive_summarizer = abstractive_summarization.AbstractiveSummarizer()

    def summarization_process_selector(self, method: int, input_path: str, evaluate: bool):
        """
        Selects and executes a summarization method based on the provided method identifier. Depending on the method
        chosen, the function applies summarization techniques to the text, saves the results to a CSV file, and logs
        the outcome.
        :param method: An integer representing the summarization method to execute. The options are:
                1: Apply TextRank for extractive summarization and save the results to a CSV file.
                2: Apply BERT for extractive summarization and save the results to a CSV file.
                3: Apply BART for abstractive summarization and save the results to a CSV file.
                4: Apply Llama for abstractive summarization and save the results to a CSV file.
        :param input_path: The path to the CSV file to be processed.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :return: The function performs text summarization and saves the results to a CSV file.
        """
        extracted_df = pd.DataFrame()
        method_name = ''

        # Load the input dataframe
        logger.info("Loading input data...")
        df_to_process = util_data_loader.load_csv_to_df(input_path)
        df_to_process = util_preprocessing.create_subset_based_on_proportions(df_to_process)

        # Extract 'text' values from the parsed dictionaries in the 'sections' column
        logger.info("Extracting 'text' values from the 'sections' column...")
        text_data = util_preprocessing.extract_text_from_sections(df_to_process)

        n_sent = 10

        # Determine the processing method and execute the corresponding segmentation technique.
        logger.info("Start summarization process...")
        match method:
            case 1:
                method_name = 'TextRank'
                summary_list = self.textrank_summarizer.apply_textrank(text_data, evaluate, n_sent=n_sent)
            case 2:
                method_name = 'LSA'
                summary_list = self.lsa_summarizer.apply_lsa(text_data, evaluate)  # TODO: Fix
            case 3:
                method_name = 'BART'
                summary_list = self.abstractive_summarizer.apply_bart(text_data, evaluate)  # TODO: Implement
            case 4:
                method_name = 'Llama3_1-8B-instruct'
                summary_list = self.abstractive_summarizer.apply_llama(text_data, evaluate)  # TODO: Implement
                pass


        if extracted_df is not None and not extracted_df.empty:
            # Generate the current timestamp
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create the filename using the method number and the current time
            filename = f"{method_name}_{current_time}.csv"
            # Combine the folder path and the filename
            file_path = os.path.join(constants.SUMMARIZATION_RESULTS_DIR, filename)
            # Save the DataFrame to the CSV file
            extracted_df.to_csv(file_path, index=False)
            logger.info(f"CSV with {method_name} saved to {file_path}!")
        else:
            error_message = "Summarization failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Summarization Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', type=int, choices=range(1, 5), default=1,
                        help=(
                            'Specify clustering method (1-5):\n'
                            '1 = TextRank (extractive summarization),\n'
                            '2 = BERT (extractive summarization),\n'
                            '3 = BART (abstractive summarization),\n'
                            '4 = Llama3.1-8B:instruct (abstractive summarization).'
                        ))
    parser.add_argument('--input', type=str, default=constants.SECTIONS_PATH.format(year=2021),  # TODO: Make dynamic
                        help="The path to the input data CSV file")
    parser.add_argument('--eval', action='store_true', help="If true, returns the evaluation scores in a CSV file.")

    args = parser.parse_args()

    # Initialize the summarization pipeline object and run the summarization process
    logger.info("Start summarization pipeline...")
    summarization_pipe = SummarizationPipeline()
    summarization_pipe.summarization_process_selector(args.method, args.input, args.eval)

    logger.info("Summarization pipeline successfully finished!")
