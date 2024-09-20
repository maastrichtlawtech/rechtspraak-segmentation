import pandas as pd
import numpy as np
import re

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from utils import constants
from typing import List
from tqdm import tqdm

# Enable the tqdm progress bar for pandas
tqdm.pandas()


def safe_tokenize(text: str) -> List[str]:
    """ Performs tokenization if text is a string, if text is any other type it returns an empty list.  """
    if isinstance(text, str):
        return sent_tokenize(text)
    else:
        return []


def tokenize_sentences(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Applies sentence tokenization to each text in the specified column of a DataFrame.
    :param df: The input dataframe with text that need to be tokenized.
    :param column_name: The name of the column to be tokenized.
    :return A dataframe with added column that contains tokenized text.
    """
    df[constants.TOKENIZED_COL] = df[column_name].progress_apply(safe_tokenize)
    return df


def generate_trigrams(sentences: List[str]) -> List[List[str]]:
    """
    Generates trigrams (lists of 3 consecutive sentences) from a list of sentences.

    If the list contains fewer than 3 sentences, returns the entire list as a single element list.
    If the list contains exactly 3 sentences, returns a list with one element containing those 3 sentences.
    If the list contains more than 3 sentences, returns a list of trigrams. If the number of sentences
    is not a multiple of 3, the last element will contain the remaining sentences.

    Args:
        sentences (List[str]): A list of sentences.

    Returns:
        List[List[str]]: A list of trigrams, where each trigram is a list of 3 sentences. The last element
        may contain fewer than 3 sentences if the total number of sentences is not divisible by 3.
    """
    n = len(sentences)

    # If there are less than 3 sentences, return list of all sentences
    if n < 3:
        return [sentences]

    trigrams = []

    # Exactly 3 sentences
    if n == 3:
        trigrams.append(sentences[:3])
    else:
        # More than 3 sentences
        i = 0
        while i < n:
            # Keep adding lists of 3 strings to the trigrams list
            if i + 3 <= n:
                trigrams.append(sentences[i:i+3])
                i += 3
            else:
                # When at the last 1 - 3 sentences, we append them together
                trigrams.append(sentences[i:])
                break

    return trigrams


def create_subset_based_on_proportions(df: pd.DataFrame, subset_size: int = 100) -> pd.DataFrame:
    """
    Creates a subset of the DataFrame based on the proportions of values in the 'instantie' column.
    Rows where constants.INHOUD_COL has NaN values are removed before selecting the subset.

    :param df: The original DataFrame.
    :param subset_size: The desired number of rows in the subset.
    :return: A subset DataFrame with proportions reflecting those in the 'instantie' column.
    """
    # Remove rows where 'constants.INHOUD_COL' has NaN values
    if constants.FULLTEXT_COL in df.columns:
        df = df.dropna(subset=[constants.INHOUD_COL, constants.FULLTEXT_COL])
    else:
        df = df.dropna(subset=[constants.INHOUD_COL])

    # Calculate the proportions of each value in the 'instantie' column
    proportions = df[constants.INSTANTIE_COL].value_counts(normalize=True)

    # Create a list to hold the subset DataFrame rows
    subset_dfs = []

    # Generate the subset based on proportions
    for value, proportion in proportions.items():
        value_subset = df[df[constants.INSTANTIE_COL] == value]
        n_samples = max(1, int(proportion * subset_size))  # Ensure at least one sample is taken
        subset_dfs.append(value_subset.sample(n=n_samples))

    # Concatenate the subset DataFrames
    subset_df = pd.concat(subset_dfs).reset_index(drop=True)

    return subset_df


def extract_texts(section_dict: dict):
    """
    Extract 'text' values from a dictionary of sections.
    :param section_dict: A dictionary containing section information.
    :return: A list of 'text' values from the sections dictionary.
    """
    return [section_dict[key]['text'] for key in section_dict if 'text' in section_dict[key]]


def extract_text_from_sections(df: pd.DataFrame):
    """
    Apply the extract_texts function to the 'sections' column in the DataFrame.
    :param df: A pandas DataFrame containing a 'sections' column with dictionaries.
    :return: A DataFrame where the 'sections' column contains lists of 'text' values.
    """
    return df['sections'].apply(extract_texts).tolist()


def combine_texts_by_label(input_df):
    """
    Combines the 'text' field for each class label within each row and stores the combined texts as a list of strings.

    Parameters:
    input_df (pd.DataFrame): The input dataframe containing the nested dictionaries with 'class' and 'text'.

    Returns:
    pd.DataFrame: The updated dataframe with a new column 'combined_texts' that contains a list of combined texts per row.
    """

    # Create a new column to store the combined texts
    combined_texts_list = []

    # Iterate over the rows in the input dataframe
    for idx, row in input_df.iterrows():
        # Extract the 'sections' dictionary
        sections = row['llm_response']

        # Create a dictionary to hold combined texts for this specific row, per class
        combined_texts_per_class = {}

        # Loop over the inner dictionaries in 'sections' (e.g., '2', '3', etc.)
        for key, value in sections.items():
            # Extract the class label and the text
            class_label = value['class']
            text = value['text']

            # Combine the text for each class label within the row
            if class_label in combined_texts_per_class:
                combined_texts_per_class[class_label].append(text)
            else:
                combined_texts_per_class[class_label] = [text]

        # For each class, join the list of texts into a single string
        combined_texts_for_row = [' '.join(texts) for texts in combined_texts_per_class.values()]

        # Add the combined texts (as a list) for this row to the list
        combined_texts_list.append(combined_texts_for_row)

    # Assign the new column to the dataframe
    input_df['combined_texts'] = combined_texts_list

    return input_df


def clean_tokenized(tokenized_text):
    # Extended cleaning to include tabs, multiple newlines, and other whitespace characters
    return [re.sub(r'\s+', ' ', re.sub(r'[^\w\s]|[\d]', '', sentence.lower())).strip() for sentence in tokenized_text]


def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(" +", " ", text)
    text = re.sub(r' (?<!\S)\d+(\.\d+)+(?!\S) ', '', text)
    text = re.sub(r'(?<!\S)\d+(\.\d{1})?\.?(?!\S)', '', text)
    text = text.strip()
    return text


def remove_stopwords(tokenized_text):
    # import stopwords from NLTK
    stop_words = stopwords.words('dutch')
    # remove the stopwords and keep remaining tokens
    sentence_tokens = [[words for words in sentence.split(' ') if words not in stop_words]
                       for sentence in tokenized_text]
    return sentence_tokens


def get_embeddings(tokens):
    for sublist in tokens:
        # Use a list comprehension to filter out empty strings from each sublist
        sublist[:] = [item for item in sublist if item != '']

    w2v = Word2Vec(tokens, vector_size=1, min_count=1, epochs=1000)
    # Initialize an empty list to store sentence embeddings
    sentence_embeddings = []

    # Calculate the sentence embeddings
    for words in tokens:
        word_embeddings = [w2v.wv[word] for word in words if word in w2v.wv]
        if word_embeddings:
            # If there are word embeddings for the words in the sentence
            sentence_embedding = np.mean(word_embeddings, axis=0)
            sentence_embeddings.append(sentence_embedding)
        # else:
        # Handle the case where no word embeddings are found for the sentence
        # You can choose to skip or assign a default value here
        #    sentence_embeddings.append(np.zeros(w2v.vector_size))

    # sentence_embeddings=[[w2v[word][0] for word in words] for words in tokens]
    # max_len=max([len(tokens) for tokens in tokens])
    # sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

    return sentence_embeddings
