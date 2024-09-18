import multiprocessing
import bs4
import pandas as pd
import os
import re

from collections import defaultdict
from bs4 import BeautifulSoup

from utils import constants, logger_script, util_data_loader

logger = logger_script.get_logger(constants.EXTRACTION_LOGGER_NAME)

class SectionExtractor:
    """
    A class dedicated to extracting sections, metadata, and structured text information from legal case documents
    in XML format, particularly from Rechtspraak case documents.

    Methods:
        replacer(match: re.Match) -> str:
            Concatenates the matched groups by excluding spaces and formatting them with "artikel" as a prefix.

        text_sectioning(doc: str, split_patterns: list[str], merge_patterns: list[str]) -> list[str]:
            Segments a document into sections based on split and merge patterns, returning a list of relevant sections.

        extract_text_from_section(section: bs4.element.Tag) -> str:
            Extracts and concatenates all stripped strings from a given HTML/XML section.

        process_xml(xml_file: str) -> list[str] or None:
            Processes a single XML file to extract metadata and key sections, returning them as a list of strings,
            or None if critical sections are missing.

        process_files_in_parallel(files: list[str]) -> list[list[str]]:
            Processes multiple XML files in parallel using multiprocessing, returning a list of results,
            each containing metadata and sections from a document.

        organize_by_number(strings: list[str]) -> dict[int, str]:
            Organizes a list of strings by their leading numbers and concatenates them into a dictionary
            with ordered numeric keys.

        extract_sections(input_path: str) -> pd.DataFrame:
            Extracts sections from XML files within a directory and organizes them into a pandas DataFrame.
    """

    # Function to concatenate the matched groups (excluding the space)
    def replacer(self, match: re.Match) -> str:
        """ Concatenates the matched groups by excluding spaces and formatting them in a specific way. """
        return f"artikel{match.group(1)}"

    # Your text_sectioning function
    def text_sectioning(self, doc: str, split_patterns: list[str], merge_patterns: list[str]) -> list[str]:
        """
        Segments a document into sections based on provided split and merge patterns.

        :param doc: The document text to be segmented.
        :param split_patterns: List of regex patterns used to split the text into sections.
        :param merge_patterns: List of regex patterns used to merge certain sections.
        :return: A list of strings representing the segmented sections of the document,
                         filtered to include only sections matching a numeric pattern.
        """
        # Apply the merge pattern to the document text, using the replacer function
        merge_patterns = merge_patterns[0]
        text = re.sub(merge_patterns, self.replacer, doc)

        # Combine split patterns into a single regex pattern
        super_pattern = "|".join(split_patterns)

        # Split the document text into sections based on the combined pattern
        sectioning = re.split(super_pattern, text)

        stripped_list = [s.lstrip() for s in sectioning]

        # Filter sections to keep only those starting with a numeric pattern (e.g., "1.")
        filtered_list = [s for s in stripped_list if re.match(r'\d+\.', s)]
        return filtered_list

    # Function to extract text without XML tags
    def extract_text_from_section(self, section: bs4.element.Tag) -> str:
        """
        Extracts and returns the text content from a section while stripping XML/HTML tags.
        :param section: A BeautifulSoup tag object representing an XML/HTML section.
        :return: Cleaned text from the section with all XML/HTML tags removed.
        """
        # Join all content within the section as a single string
        section_text = ''.join(str(child) for child in section.contents)
        # Create a new BeautifulSoup object to parse the content and strip tags
        clean_text = BeautifulSoup(section_text, 'html.parser').get_text()
        return clean_text

    def process_xml(self, xml_file: str) -> list[str]:
        """
        Processes a single XML file to extract legal document information and its sections.
        :param xml_file: Path to the XML file to be processed.
        :return: A list containing extracted metadata and the text content of various sections.
                  Returns None if critical sections are missing.
        """
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')

            # Initialize variables
            procesverloop_text, overwegingen_text, beslissing_text = '', '', ''
            ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing = '', '', '', '', '', ''

            # Extract global information
            ecli_tag = soup.find("dcterms:identifier")
            date_tag = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"})
            inhoud_tag = soup.find("inhoudsindicatie")
            legal_body_tag = soup.find("dcterms:creator", {"rdfs:label": "Instantie"})
            rechtsgebied_tag = soup.find("dcterms:subject", {"rdfs:label": "Rechtsgebied"})
            wetsverwijzing_tag = soup.find("dcterms:references", {"rdfs:label": "Wetsverwijzing"})

            if ecli_tag: ecli = ecli_tag.text
            if date_tag: date = date_tag.text
            if inhoud_tag: inhoud = inhoud_tag.text
            if legal_body_tag: legal_body = legal_body_tag.text
            if rechtsgebied_tag: rechtsgebied = rechtsgebied_tag.text
            if wetsverwijzing_tag: wetsverwijzing = wetsverwijzing_tag.text

            # Process each section and convert to pure text
            sections = soup.find_all("section")
            for sec in sections:
                role = sec.get('role')

                # Use a common method to extract plain text
                section_text = self.extract_text_from_section(sec)

                # Append text based on the role of the section
                if role == 'procesverloop':
                    procesverloop_text += (' ' + section_text if procesverloop_text else section_text)
                elif role == 'beslissing':
                    beslissing_text += (' ' + section_text if beslissing_text else section_text)
                else:  # This includes 'overwegingen' or sections without a role
                    overwegingen_text += (' ' + section_text if overwegingen_text else section_text)

            # Check if critical sections are present
            if not procesverloop_text or not beslissing_text:
                return None  # Skip file if critical sections are missing

            # Compile all extracted information into a list
            judgement_list = [ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing, procesverloop_text, overwegingen_text, beslissing_text]
            return judgement_list


    def process_files_in_parallel(self, files: list[str]) -> list[list[str]]:
        """
        Processes multiple XML files in parallel using multiprocessing.
        :param files: List of paths to XML files.
        :return: A list of results where each result is the output of `process_xml`.
        """
        # Get the number of CPU cores to use for multiprocessing
        num_processes = multiprocessing.cpu_count()
        logger.info(f"Multiprocessing with {num_processes}.")

        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)

        # Use the multiprocessing pool to process the XML files in parallel
        logger.info("start multiprocessing here")
        result_lists = pool.map(self.process_xml, files)
        logger.info(f"Number of files processed: {len(result_lists)}")

        # Close the pool and wait for the worker processes to finish
        pool.close()
        pool.join()

        return result_lists

    @staticmethod
    def organize_by_number(strings: list[str]) -> dict[int, str]:
        """
        Organizes a list of strings based on their leading number and concatenates them.
        :param strings: A list of strings to organize by the number before the first period.
        :return: A dictionary where the keys are the leading numbers (as integers),
                  and the values are the concatenated strings associated with that number.
        """
        # Initialize a defaultdict to hold the organized strings
        result = defaultdict(list)

        # Loop through each string and extract the leading number
        for string in strings:
            period_index = string.find('.')
            if period_index != -1:
                number = string[:period_index]
                if number.isdigit():
                    result[int(number)].append(string)

        # Join the lists into a single string per key
        for key in result:
            result[key] = ' '.join(result[key])

        # Sort the dictionary by keys to ensure order
        sorted_result = dict(sorted(result.items()))
        return sorted_result

    def extract_sections(self, input_path: str) -> pd.DataFrame:
        """
        Extracts sections from XML files in the input directory, processes them,
        and organizes them into a pandas DataFrame.
        :param input_path: Path to the directory containing the XML files.
        :return: A DataFrame containing the extracted metadata and sections from the XML files.
        """
        # Find all XML files in the input directory
        xml_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.xml')]

        # Process the XML files in parallel
        result_lists = self.process_files_in_parallel(xml_files)
        filtered_results = [result for result in result_lists if result is not None]

        # Create df for metadata
        column_names = ['ecli', 'date', 'inhoudsindicatie', 'instantie',
                        'rechtsgebied', 'wetsverwijzing', 'procesverloop',
                        'overwegingen', 'beslissing']
        df = pd.DataFrame(filtered_results, columns=column_names)

        # Load the split and merge patterns for text sectioning
        documents = df[constants.OVERWEGINGEN_COL].tolist()
        split_patterns = util_data_loader.load_txt_file(constants.SPLIT_PATTERNS_PATH)
        merge_patterns = util_data_loader.load_txt_file(constants.MERGE_PATTERNS_PATH)

        segmented = []
        # Segment the text of each document
        for i, text in enumerate(documents):
            if i % 100 == 0:
                logger.info(f"Processing document number: {i} of {len(documents)}")
            # Apply the text sectioning method to each document
            sections = self.text_sectioning(text, split_patterns, merge_patterns)
            sections = [item for item in sections if item is not None]
            segmented.append(sections)

        # Add the segmented sections to the DataFrame
        df[constants.SECTIONS_COL] = segmented

        # Organize the sections by leading number
        df[constants.SECTIONS_COL] = df[constants.SECTIONS_COL].apply(self.organize_by_number)

        return df

