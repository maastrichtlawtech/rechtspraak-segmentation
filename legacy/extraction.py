import rechtspraak_extractor as rex
import pandas as pd
import os
import multiprocessing
from bs4 import BeautifulSoup

from utils import process_files_in_parallel

# METADATA DF EXTRACTION

def xml_to_metadata_extraction(input_path: str) -> pd.DataFrame:
    """
    Extracts sections from XML files in the input directory, processes them,
    and organizes them into a pandas DataFrame.
    :param input_path: Path to the directory containing the XML files.
    :return: A DataFrame containing the extracted metadata and sections from the XML files.
    """
    # Find all XML files in the input directory
    xml_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.xml')]

    # Process the XML files in parallel
    result_lists = process_files_in_parallel(xml_files, process_xml_metadata)
    filtered_results = [result for result in result_lists if result is not None]

    # Create df for metadata
    column_names = ['ecli', 'date', 'inhoudsindicatie', 'instantie',
                    'rechtsgebied', 'wetsverwijzing']
    df = pd.DataFrame(filtered_results, columns=column_names)

    return df

def process_xml_metadata(xml_file: str) -> list[str]:
    """
    Processes a single XML file to extract legal document information and its sections.
    :param xml_file: Path to the XML file to be processed.
    :return: A list containing extracted metadata and the text content of various sections.
                Returns None if critical sections are missing.
    """
    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')

        # Initialize variables
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

        # Compile all extracted information into a list
        judgement_list = [ecli, date, inhoud, legal_body, rechtsgebied, wetsverwijzing]
        return judgement_list

# FULLTEXT EXTRACTION

def xml_to_fulltext_dict(input_path):
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
    result_dict_lists = process_files_in_parallel(xml_files, process_xml_fulltext)

    # Define the column names for the resulting DataFrame and create a DataFrame from the extracted results
    return result_dict_lists

def process_xml_fulltext(xml_file):
    """
    Processes a single XML file to extract relevant legal judgement information, including the ECLI, date,
    inhoudsindicatie, and full text of the judgement.
    :param xml_file: The path to the XML file to be processed.
    :return: A list containing the extracted 'ecli', 'date', 'inhoudsindicatie', and 'fulltext' values.
    """
    # Open the XML file and parse it with BeautifulSoup
    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')  # Adjust the parser based on your XML format

        # Extract the ECLI, date, and inhoudsindicatie
        ecli = soup.find("dcterms:identifier").get_text() if soup.find("dcterms:identifier") else ''

        # Extract the text from all <section> tags and combine it
        sections = soup.find_all('section')
        combined_text = ' '.join(section.get_text(separator=' ', strip=True) for section in sections)

        # Append the extracted information to the judgement_list
        result_dict = {
            'ecli': ecli,
            'text': combined_text
        }

    return result_dict
