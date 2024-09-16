import os
import argparse
import pandas as pd
from datetime import datetime

from data_processing import header_extraction, full_text_extraction
from utils import constants, logger_script

logger = logger_script.get_logger(constants.EXTRACTION_LOGGER_NAME)


class DataProcessing:
    """
        A class dedicated to extract useful data from Rechtspraak case documents and saves them in a CSV.

    Attributes:
        full_text_extractor (FullTextExtractor): Extracts the full text content from documents.
        header_extractor (HeaderExtractor): Extracts header information additional to sections from documents.

    Methods:
        data_process_selector(method: int):
            Selects and applies the appropriate data processing method based on the given method number.
            - method 1: Full Text Extraction
            - method 2: Header Extraction
    """

    def __init__(self):
        """
        Initializes the data processing class with specific components for extracting data from the raw XML files.
        """
        self.full_text_extractor = full_text_extraction.FullTextExtractor()
        self.header_extractor = header_extraction.HeaderExtractor()

    def data_process_selector(self, method: int, input_path: str):
        """
        Selects and executes a data processing method based on the provided method identifier. Depending on the method
        chosen, the function extracts different types of data, saves the results to a CSV file, and logs the outcome.
        :param method: An integer representing the data processing method to execute. The options are:
            1: Extracts full text and saves it to a CSV file at the specified path.
            2: Extracts headers additional to sections and saves them to a CSV file at the specified path.
        :param input_path: The path to the file to be processed.
        :return: The function performs data extraction and saves the results to a CSV file.
        """
        extracted_df = pd.DataFrame()
        method_name = ''

        logger.info("Start header extraction process...")
        match method:
            case 1:
                method_name = 'fulltext'
                extracted_df = self.full_text_extractor.extract_fulltext(input_path)
            case 2:
                method_name = 'headers'
                extracted_df = self.header_extractor.extract_headers(input_path)

        if extracted_df is not None and not extracted_df.empty:
            # Generate the current timestamp TODO: change this naming to the year instead of datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create the filename using the method name and the current time
            filename = f"{method_name}_{current_time}.csv"
            # Combine the folder path and the filename
            file_path = os.path.join(constants.METADATA_DIR, filename)
            # Save the DataFrame to the CSV file
            extracted_df.to_csv(file_path, index=False)
            logger.info(f"CSV with {method_name} saved to {file_path}!")
        else:
            error_message = "Data extraction failed: the resulting DataFrame is empty or None."
            logger.error(error_message)
            raise ValueError(error_message)


if __name__ == '__main__':
    # create the argument parser, add the arguments
    parser = argparse.ArgumentParser(description='Rechtspraak Data Processing',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', type=int, choices=range(1, 3), default=3,
                        help=(
                            'Specify processing method (1-3): \n'
                            '1 = Full Text Extraction: creates a dataframe with a column that contains the document '
                            'full text (composed from "procesverloop", "overwegingen", and "beslissing"), \n'
                            '2 = Header Extraction: creates a dataframe with a column that holds a dictionary with '
                            'section header and section text. '
                        ))
    parser.add_argument('--input', type=str, default=constants.RAW_DIR.format(year=2021),
                        help="The path to the input data CSV file")

    args = parser.parse_args()

    # Initialize the data processing object and process data
    logger.info("Start extraction pipeline...")
    data_processor = DataProcessing()
    data_processor.data_process_selector(args.method, args.input)

    logger.info("Extraction pipeline successfully finished!")
