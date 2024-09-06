import os
import pandas as pd
from bs4 import BeautifulSoup
import multiprocessing

from tqdm import tqdm
from utils import constants, logger_script

logger = logger_script.get_logger(constants.EXTRACTION_LOGGER_NAME)


class FullTextExtractor:
    """
    A class dedicated to extracting full text from Dutch legal case XML files and processing them into a structured
    format.

    Methods:
        is_valid_tag(tag: bs4.element.Tag) -> bool:
            Checks if a given tag is valid for processing (i.e., not a 'title' tag).
        process_xml(self, xml_file: str) -> list:
            Processes a single XML file to extract relevant legal judgement information, including the ECLI, date,
            inhoudsindicatie, and full text.
        process_files_in_parallel(self, files: list) -> list:
            Processes a list of XML files in parallel, extracting the legal judgement information from each file.
        extract_fulltext(self, input_path: str) -> pd.DataFrame:
            Extracts legal judgement information from all XML files in a specified directory, converts the extracted
            data into a DataFrame, and returns the DataFrame.
    """

    @staticmethod
    def is_valid_tag(tag):
        """
        Checks if a given tag is valid for processing, specifically ensuring it is not a 'title' tag.
        :param tag: A BeautifulSoup tag object representing an XML/HTML tag.
        :return: True if the tag is valid for processing, False otherwise.
        """
        return tag.name != 'title'

    @staticmethod
    def process_xml_for_fulltext(xml_file):
        """
        Processes a single XML file to extract relevant legal judgement information, including the ECLI, date,
        inhoudsindicatie, and full text of the judgement.
        :param xml_file: The path to the XML file to be processed.
        :return: A list containing the extracted 'ecli', 'date', 'inhoudsindicatie', and 'fulltext' values.
        """
        judgement_list = []

        # Open the XML file and parse it with BeautifulSoup
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')  # Adjust the parser based on your XML format

            # Extract the ECLI, date, and inhoudsindicatie
            ecli = soup.find("dcterms:identifier").get_text() if soup.find("dcterms:identifier") else ''
            date = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"}).get_text() \
                if soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"}) else ''
            inhoud = soup.find("inhoudsindicatie").get_text() if soup.find("inhoudsindicatie") else ''
            instantie = soup.find("dcterms:creator", {"rdfs:label": "Instantie"}).get_text() \
                if soup.find("dcterms:creator", {"rdfs:label": "Instantie"}) else ''

            # Extract the text from all <section> tags and combine it
            sections = soup.find_all('section')
            combined_text = ' '.join(section.get_text(separator=' ', strip=True) for section in sections)

            # Append the extracted information to the judgement_list
            judgement_list.extend([ecli, date, inhoud, instantie, combined_text])

        return judgement_list

    def process_files_in_parallel(self, files):
        """
        Processes a list of XML files in parallel, utilizing all available CPU cores, to extract legal judgement
        information.
        :param files: A list of paths to the XML files to be processed.
        :return: A list of lists, where each sublist contains the extracted data from one XML file.
        """
        # Determine the number of CPU cores available for parallel processing
        num_processes = multiprocessing.cpu_count()

        logger.info(f"Start multiprocessing XML documents for full texts with {num_processes} processes...")
        # Initialize the multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap_unordered to process files as they complete
            # Wrap the iterator with tqdm for a progress bar
            result_iter = pool.imap_unordered(self.process_xml_for_fulltext, files)
            result_lists = list(tqdm(result_iter, total=len(files), desc="Processing XML files for full texts"))

        logger.info("Extraction of full texts finished!")
        # Return the list of results from all processed files
        return result_lists

    def extract_fulltext(self, input_path):
        """
        Extracts legal judgement information from all XML files in a specified directory, including ECLI, date,
        inhoudsindicatie, and full text, and converts the extracted data into a Pandas DataFrame.
        :param input_path: The path to the directory containing the XML files.
        :return: A Pandas DataFrame containing the extracted 'ecli', 'date', 'inhoudsindicatie', and 'fulltext' for
            each document.
        """
        # Get a list of all XML files in the specified directory
        xml_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.xml')]

        # Process the files in parallel and retrieve the results
        result_lists = self.process_files_in_parallel(xml_files)

        # Define the column names for the resulting DataFrame and create a DataFrame from the extracted results
        column_names = [constants.ECLI_COL,
                        constants.DATE_COL,
                        constants.INHOUD_COL,
                        constants.INSTANTIE_COL,
                        constants.FULLTEXT_COL]
        result_df = pd.DataFrame(result_lists, columns=column_names)

        return result_df
