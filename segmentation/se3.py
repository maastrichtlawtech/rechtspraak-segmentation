import pandas as pd
import torch

from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import constants, logger_script, util_preprocessing
from tqdm import tqdm
from typing import List, Dict, Union

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class Se3Clusterer:
    """
        A class to perform self-segmentation of legal documents using a Transformer models and semantic similarity.

    The Se3Clusterer applies a segmentation method where documents are split into chunks based on their semantic
    similarity. The class uses a pre-trained Transformer models to encode sentences and then assigns them to chunks that
    maximize semantic coherence.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from the pre-trained Transformer models.
        model (AutoModel): Pre-trained Transformer models used to encode sentences.

    Methods:
        semantic_similarity(sentence_embedding, chunk_embedding):
            Computes the cosine similarity between sentence and chunk embeddings.

        create_chunks(document, min_size, max_size):
            Segments the document into chunks based on the semantic similarity of sentences.

        compute_rouge(chunk, summary_sentence):
            Computes the ROUGE-1 precision score between a chunk of text and a summary sentence.

        assign_targets(chunks, summary_sentences):
            Assigns the most similar summary sentence to each chunk of text.

        process_se3_segmentation(input_df):
            Processes a DataFrame of legal documents by segmenting them into chunks and assigning summary targets.
    """

    def __init__(self):
        """
        Initializes the Se3Clusterer with a tokenizer and a pre-trained Transformer models for encoding sentences.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(constants.LEGAL_MULTILING_TF)
        self.model = AutoModel.from_pretrained(constants.LEGAL_MULTILING_TF)

    @staticmethod
    def semantic_similarity(sentence_embedding: torch.Tensor, chunk_embedding: torch.Tensor):
        """
        Computes the cosine similarity between the embeddings of a sentence and a chunk.
        :param sentence_embedding: A tensor representing the embedding of a sentence.
        :param chunk_embedding: A tensor representing the embedding of a chunk of text.
        :return: A float representing the mean cosine similarity between the sentence and chunk embeddings.
        """
        # Convert to numpy arrays
        sentence_embedding_np = sentence_embedding.detach().cpu().numpy()
        chunk_embedding_np = chunk_embedding.detach().cpu().numpy()

        # Calculate cosine similarity using sklearn
        cosine_sim = cosine_similarity(sentence_embedding_np, chunk_embedding_np)
        return cosine_sim.mean()

    def create_chunks(self, document: List[str], min_size: int, max_size: int) -> List[List[str]]:
        """
        Segments a document into chunks based on semantic similarity.
        This method iteratively adds sentences to a chunk until the chunk reaches a specified maximum size.
        If adding a sentence exceeds the maximum size, the sentence is compared to existing chunks, and the chunk
        with the highest similarity is chosen.
        :param document: A list of sentences representing the document to be segmented.
        :param min_size: The minimum size (in tokens) for a chunk.
        :param max_size: The maximum size (in tokens) for a chunk.
        :return: A list of chunks, where each chunk is a list of sentences.
        """
        chunks = []  # List to store the resulting chunks
        current_chunk = []  # Current chunk being processed

        for sentence in document:
            # Tokenize the sentence and calculate the size of the current chunk
            sentence_tokens = self.tokenizer.encode(sentence, return_tensors='pt')
            current_chunk_size = sum(len(self.tokenizer.encode(s)) for s in current_chunk)

            if current_chunk_size + len(sentence_tokens[0]) < min_size:
                # Add sentence to the current chunk if it doesn't exceed the minimum size
                current_chunk.append(sentence)
            elif current_chunk_size + len(sentence_tokens[0]) > max_size:
                # If adding the sentence exceeds the max size, finalize the current chunk and start a new one
                chunks.append(current_chunk)
                current_chunk = [sentence]
            else:
                if current_chunk:
                    # Compute the embeddings for the current chunk and the sentence
                    chunk_embedding = self.model(**self.tokenizer(current_chunk,
                                                                  return_tensors='pt',
                                                                  padding=True,
                                                                  truncation=True)).last_hidden_state.mean(dim=1)
                    sentence_embedding = self.model(**self.tokenizer(sentence,
                                                                     return_tensors='pt')).last_hidden_state.mean(dim=1)
                    # Calculate the similarity and decide where to add the sentence
                    similarity = self.semantic_similarity(sentence_embedding, chunk_embedding)
                    # Assign to the chunk with higher similarity
                    if similarity > 0.5:  # This threshold is arbitrary; adjust as needed
                        current_chunk.append(sentence)
                    else:
                        # If similarity is low, finalize the current chunk and start a new one
                        chunks.append(current_chunk)
                        chunks.append(current_chunk)
                        current_chunk = [sentence]
        if current_chunk:
            # Add the last chunk if it exists
            chunks.append(current_chunk)
        return chunks

    @staticmethod
    def compute_rouge(chunk: List[str], summary_sentence: str) -> float:
        """
        Computes the ROUGE-1 precision score between a chunk of text and a summary sentence.
        :param chunk: A list of sentences representing a chunk of text.
        :param summary_sentence: A string representing a summary sentence.
        :return: A float representing the ROUGE-1 precision score between the chunk and the summary sentence.
        """
        # Instantiate a ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        # Compute ROUGE-1 precision score
        scores = scorer.score(' '.join(chunk), summary_sentence)
        rouge1_precision = scores['rouge1'].precision

        return rouge1_precision

    def assign_targets(self, chunks: List[List[str]], summary_sentences: List[str]) -> List[Union[str, None]]:
        """
        Assigns the most similar summary sentence to each chunk of text based on ROUGE-1 precision scores.
        :param chunks: A list of chunks, where each chunk is a list of sentences.
        :param summary_sentences: A list of summary sentences to be matched to the chunks.
        :return: A list of summary sentences, each corresponding to the most similar chunk.
        """
        targets = []  # List to store the best matching summary for each chunk
        for chunk in chunks:
            best_target = None
            best_rouge_score = 0
            for summary_sentence in summary_sentences:
                # Compute the ROUGE score between the chunk and the summary sentence
                rouge_score = self.compute_rouge(chunk, summary_sentence)  # Implement this function
                if rouge_score > best_rouge_score:
                    # Update the best match if this score is higher than previous scores
                    best_rouge_score = rouge_score
                    best_target = summary_sentence
            targets.append(best_target)  # Append the best match or None if no match found
        return targets

    def process_se3_segmentation(self,
                                 input_df: pd.DataFrame,
                                 evaluate: bool) -> List[Dict[str, Union[List[str], List[Union[str, None]]]]]:
        """
        Processes a DataFrame of legal documents by segmenting them into chunks and assigning summary targets.
        This method first creates a subset of the input DataFrame based on the proportions of values in the 'instantie'
        column. Then, it splits each document into chunks and assigns the most similar summary sentence to each chunk.
        The results are returned as a list of dictionaries containing the chunks and their corresponding targets.
        :param input_df: A pandas DataFrame containing the documents to be processed.
        :param evaluate: A bool that indicates whether evaluation scores must be saved.
        :return: A list of dictionaries, where each dictionary contains 'chunks' (list of sentences) and 'targets'
            (list of summary sentences).
        """
        results = []  # List to store the final segmentation results
        min_size, max_size = 64, 128  # Define the minimum and maximum chunk sizes

        # Create a subset of the DataFrame based on the proportions of 'instantie'
        subset_df = util_preprocessing.create_subset_based_on_proportions(input_df)

        total_rouge_score = 0
        n = 0  # To track the number of comparisons

        for index, row in tqdm(subset_df.iterrows(), total=subset_df.shape[0], desc="Processing documents with Se3"):
            document = row[constants.FULLTEXT_COL].split('. ')  # Split document into sentences
            summary_sentences = row[constants.INHOUD_COL].split('. ')  # Split summary into sentences

            # Create chunks from the document
            chunks = self.create_chunks(document, min_size, max_size)

            # Assign the most relevant summary sentences to the chunks
            targets = self.assign_targets(chunks, summary_sentences)

            # Evaluate ROUGE score between chunks and summary sentences
            if evaluate:
                for chunk, target in zip(chunks, targets):
                    if target is not None:
                        rouge_score = self.compute_rouge(chunk, target)
                        total_rouge_score += rouge_score
                        n += 1

            # Store the results
            results.append({
                'chunks': chunks,
                'targets': targets
            })

        if evaluate:
            # Compute the average ROUGE score across all documents
            avg_rouge_score = total_rouge_score / n if n > 0 else 0
            logger.info(f"Average ROUGE-1 Precision Score: {avg_rouge_score:.4f}")

        return results
