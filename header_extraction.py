import os
import bs4
import logging
import pandas as pd
from bs4 import BeautifulSoup

from utils import constants, logger_script
from tqdm import tqdm

logger = logger_script.get_logger(constants.EXTRACTION_LOGGER_NAME)


class HeaderExtractor:
    """
    A class dedicated to extracting header information from Rechtspraak case documents and processing it into a
    structured format.

    Methods:
        extract_text_from_section(section: bs4.element.Tag) -> str:
            Extracts and concatenates all stripped strings from a given HTML/XML section.
        extract_section_info(soup: bs4.BeautifulSoup) -> dict:
            Extracts section-specific information, including section numbers, headers, and full text, from a parsed
            XML/HTML document.
        process_xml(xml_file: str) -> dict:
            Processes a single XML file to extract relevant legal judgement information, including global attributes
            and section-specific data.
        process_xml_files_in_folder(folder_path: str) -> list:
            Processes all XML files within a specified folder, extracting legal judgement information from each file
            and returning it as a list of dictionaries.
        extract_headers() -> pd.DataFrame:
            Extracts legal judgement headers from XML files in a specified folder, converts the extracted data into a
            DataFrame, and optionally saves it to a CSV file.
    """
    def __init__(self):
        """ Initializes a skip counter to log the number of files skipped. """
        self.skip_counter = 0

    @staticmethod
    def extract_text_from_section(section: bs4.Tag) -> str:
        """
        Extracts and concatenates all stripped strings from a given HTML/XML section.
        :param section: A BeautifulSoup tag object representing a section.
        :return: The concatenated string of all text elements within the section.
        """
        return ' '.join(section.stripped_strings)

    def extract_section_info(self, soup: bs4.BeautifulSoup) -> dict:
        """
        Extracts information from each section in the provided BeautifulSoup object, including section number, header
        text, and the full text of the section.
        :param soup: A BeautifulSoup object representing the parsed XML/HTML document.
        :return: A dictionary where each key is a section number and the value is another dictionary containing
              the 'header' (title of the section) and 'text' (full text of the section).
        """
        # Initialize an empty dictionary to hold section data
        section_data = {}

        # Find all 'section' tags within the soup object
        sections = soup.find_all("section")

        # Iterate over each 'section' tag found
        for sec in sections:
            # Find the 'title' tag within the current section
            title_tag = sec.find("title")
            # Check if the 'title' tag exists
            if title_tag:
                # Find the 'nr' tag within the 'title' tag, which represents the section number
                nr_tag = title_tag.find("nr")
                # Ensure the 'nr' tag exists and contains text
                if nr_tag and nr_tag.text:
                    # Extract the section number from the 'nr' tag's text
                    section_number = nr_tag.text
                    # Extract the header text, removing the section number from the title
                    header_text = ''.join(title_tag.stripped_strings).replace(nr_tag.text, '').strip()
                    # Extract the full text content of the section
                    section_text = self.extract_text_from_section(sec)
                    # Store the section number, header, and text in the dictionary
                    section_data[section_number] = {'header': header_text, 'text': section_text}

        # Return the dictionary containing all extracted section data
        return section_data

    def process_xml(self, xml_file: str) -> dict or None:
        """
        Processes a single XML file to extract relevant legal judgement information, including global attributes
        and section-specific data.
        :param xml_file: The path to the XML file to be processed.
        :return: A dictionary containing extracted information such as 'ecli', 'date', 'inhoud', 'legal_body',
              'rechtsgebied', 'wetsverwijzing', and 'sections'. Returns None if no valid sections are found.
        """
        # Open the XML file and parse it using BeautifulSoup with the 'xml' parser
        with open(xml_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')

            # Initialize variables to hold the extracted global information
            ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing = '', '', '', '', '', ''

            # Extract global information from the XML tags
            ecli_tag = soup.find("dcterms:identifier")  # Extract the ECLI identifier
            date_tag = soup.find("dcterms:date", {"rdfs:label": "Uitspraakdatum"})  # Extract the judgement date
            inhoud_tag = soup.find("inhoudsindicatie")  # Extract the summary or indication of content
            legal_body_tag = soup.find("dcterms:creator",
                                       {"rdfs:label": "Instantie"})  # Extract the legal body or court
            rechtsgebied_tag = soup.find("dcterms:subject",
                                         {"rdfs:label": "Rechtsgebied"})  # Extract the legal area or domain
            wetsverwijzing_tag = soup.find("dcterms:references",
                                           {"rdfs:label": "Wetsverwijzing"})  # Extract legal references

            # Assign extracted text values to the corresponding variables if the tags were found
            if ecli_tag:
                ecli = ecli_tag.text
            if date_tag:
                date = date_tag.text
            if inhoud_tag:
                inhoud = inhoud_tag.text
            if legal_body_tag:
                legal_body = legal_body_tag.text
            if rechtsgebied_tag:
                rechtsgebied = rechtsgebied_tag.text
            if wetsverwijzing_tag:
                wetsverwijzing = wetsverwijzing_tag.text

            # Extract section-specific information by calling the extract_section_info method
            section_data = self.extract_section_info(soup)

            # If no valid sections are found, log a debug message and skip processing this file
            if not section_data:
                return None

            # Compile all the extracted information into a dictionary
            judgement_data = {
                'ecli': ecli,
                'date': date,
                'inhoud': inhoud,
                'legal_body': legal_body,
                'rechtsgebied': rechtsgebied,
                'wetsverwijzing': wetsverwijzing,
                'sections': section_data
            }

            # Return the dictionary containing all the extracted data
            return judgement_data

    def process_xml_files_in_folder(self, folder_path: str) -> list:
        """
        Processes all XML files within a specified folder, extracting legal judgement information from each file.
        :param folder_path: The path to the folder containing XML files.
        :return: A list of dictionaries where each dictionary contains the extracted judgement data from an XML file.
        """
        # Initialize an empty list to store the extracted data from all files
        all_judgements = []
        # Initialize a counter to keep track of the number of files processed
        file_counter = 0

        # Loop over all files in the specified folder
        for filename in tqdm(os.listdir(folder_path), desc="Processing XML files for header extraction"):
            # Check if the file has a .xml extension
            if filename.endswith('.xml'):
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                # Increment the file counter
                file_counter += 1
                # Log the start of processing for the current file
                if file_counter % 10000 == 0:
                    logger.info(f"Processed {file_counter} files!")
                # Process the XML file to extract judgement data
                judgement_data = self.process_xml(file_path)
                # If data is successfully extracted, add it to the list
                if judgement_data:
                    all_judgements.append(judgement_data)
                else:
                    # Log a debug message if no valid data was extracted from the file
                    self.skip_counter += 1

        # Return the list containing the extracted data from all processed files
        return all_judgements

    def extract_headers(self, input_path: str) -> pd.DataFrame or None:
        """
        Extracts legal judgement headers from XML files in a specified folder and converts the extracted data into a
        DataFrame.
        :return: Saves the resulting DataFrame  to a CSV file specified in the constants.
        """
        # Process all XML files in the specified folder and extract judgement data
        all_judgements = self.process_xml_files_in_folder(input_path)

        logger.debug(f"Skipped {self.skip_counter}: No valid sections found for these files!")

        # Check if any judgement data was extracted
        if all_judgements:
            # Convert the list of extracted judgements into a DataFrame
            df = pd.DataFrame(all_judgements)
            return df
        else:
            # Log an informational message if no valid judgement data was found
            logger.info("No valid judgements found in the XML files.")
