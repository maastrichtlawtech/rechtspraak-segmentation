import networkx as nx
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils import constants, logger_script, util_preprocessing
from scipy import spatial
from tqdm import tqdm

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


# TODO: resolve all issues using the code from old old repo
class TextRankSummarizer:

    def __init__(self):
        pass

    @staticmethod
    def get_similarity_matrix(tokens, embeddings):
        similarity_matrix = np.zeros([len(tokens), len(tokens)])
        for i, row_embedding in enumerate(embeddings):
            for j, column_embedding in enumerate(embeddings):
                similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)
        return similarity_matrix

    def apply_textrank(self, text_data,  evaluate: bool, n_sent: int = 5):
        """
        Apply the TextRank algorithm to a list of documents (nested list of sentences).

        :param text_data: A list of lists, where each sublist contains the sentences of a document.
        :param n_sent: Number of top sentences to extract.
        :return: A list of top sentences for each document.
        """
        summarized_documents = []

        for document in tqdm(text_data, desc="Applying TextRank", unit="document"):
            # Join the sentences into a single text for tokenization
            text = " ".join(document)

            # Clean and tokenize text
            tokenized = sent_tokenize(text)
            clean_tokens = util_preprocessing.clean_tokenized(tokenized)
            no_stopwords_tokens = util_preprocessing.remove_stopwords(clean_tokens)

            # Get the sentence embeddings
            embeddings = util_preprocessing.get_embeddings(no_stopwords_tokens)

            # Get the similarity matrix
            similarity_matrix = self.get_similarity_matrix(no_stopwords_tokens, embeddings)

            # Convert similarity matrix to a graph
            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)

            # Rank sentences based on their scores
            top_sentence = {sentence: scores[index] for index, sentence in enumerate(tokenized)}
            top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:n_sent])

            # Collect top-ranked sentences in the order they appeared in the text
            top_sentences = []
            for sent in tokenized:
                if sent in top.keys():
                    top_sentences.append(sent)

            # Append the summarized text for this document to the results
            summarized_documents.append(' '.join(top_sentences))

        return summarized_documents


class LSAExtractiveSummarizer:
    def __init__(self, n_components=1):
        """
        Initialize the LSA summarizer with the number of components for SVD.

        :param n_components: Number of components to keep after performing SVD.
                             Typically 1 or 2 is used for summarization.
        """
        self.n_components = n_components

    def filter_short_sentences(self, sentences, min_length=10):
        """
        Filter out sentences that are too short or contain mostly punctuation/numbers.

        :param sentences: List of sentences to filter.
        :param min_length: Minimum number of characters a sentence must have to be considered.
        :return: List of filtered sentences and their original indices.
        """
        filtered_sentences = []
        original_indices = []
        for i, sentence in enumerate(sentences):
            # Filter out sentences that are too short or consist mostly of punctuation/numbers
            if len(sentence) >= min_length and any(char.isalpha() for char in sentence):
                filtered_sentences.append(sentence)
                original_indices.append(i)

        # Debugging: Print filtered sentences
        print(f"Filtered Sentences: {filtered_sentences}")
        return filtered_sentences, original_indices

    def lsa_summarization(self, sentences, top_n=3):
        """
        Perform LSA-based summarization by applying SVD on the TF-IDF matrix of sentences.

        :param sentences: List of sentences from the document.
        :param top_n: Number of top sentences to extract for the summary.
        :return: List of top N sentences forming the summary.
        """
        # Filter out short sentences
        filtered_sentences, original_indices = self.filter_short_sentences(sentences)

        # Return empty summary if no valid sentences remain
        if len(filtered_sentences) == 0:
            return []

        # Step 1: Convert sentences to TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary size for efficiency
        tfidf_matrix = vectorizer.fit_transform(filtered_sentences)

        # Debugging: Check the shape of the TF-IDF matrix
        print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

        # Step 2: Perform SVD (Latent Semantic Analysis)
        svd = TruncatedSVD(n_components=self.n_components, n_iter=100)
        svd_matrix = svd.fit_transform(tfidf_matrix)

        # Debugging: Check the shape and contents of the SVD matrix
        print(f"SVD Matrix Shape: {svd_matrix.shape}")
        print(f"SVD Matrix (First Component): {svd_matrix[:, 0]}")

        # Step 3: Rank sentences based on their contribution to the first principal component
        ranked_sentences = np.argsort(svd_matrix[:, 0])[::-1]  # Sort by the first component in descending order

        # Step 4: Select the top N sentences
        top_sentence_indices = ranked_sentences[:top_n]

        # Map back to original sentence indices
        top_sentences = [sentences[original_indices[i]] for i in top_sentence_indices]

        # Debugging: Print the selected top sentences
        print(f"Top Sentences: {top_sentences}")

        return top_sentences

    def apply_lsa(self, documents, top_n=3):
        """
        Apply LSA summarization to a list of documents.

        :param documents: List of documents, each document is a string of text.
        :param top_n: Number of top sentences to extract for each document summary.
        :return: List of summaries (one summary per document).
        """
        summarized = []

        for document in tqdm(documents, desc="Applying LSA Summarization", unit="document"):
            # Join the sentences into a single text for tokenization
            document = " ".join(document)

            # Step 1: Tokenize the document into sentences
            sentences = sent_tokenize(document)

            # Step 2: Skip empty documents
            if len(sentences) == 0:
                print("Document has no sentences, skipping.")  # Debugging line
                continue  # Skip empty documents

            # Step 3: Perform LSA summarization on the tokenized sentences
            top_sentences = self.lsa_summarization(sentences, top_n)
            summarized.append(top_sentences)

        return summarized