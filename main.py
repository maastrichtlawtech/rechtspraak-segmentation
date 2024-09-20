import argparse
import logging
from utils import constants
from data_processing.data_processing_pipeline import DataProcessing
from segmentation.segmentation_pipeline import SegmentationPipeline
from summarization.summarization_pipeline import SummarizationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rechtspraak Unified Pipeline',
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    subparsers = parser.add_subparsers(dest='pipeline', help='Select the pipeline to run')
    
    # Data Processing pipeline arguments
    dp_parser = subparsers.add_parser('data_processing', help='Run the data processing pipeline')
    dp_parser.add_argument('--method', type=int, choices=range(1, 4), default=3,
                        help=(
                            'Specify processing method (1-3): \n'
                            '1 = Full Text Extraction: creates a dataframe with a column that contains the document '
                            'full text (composed from "procesverloop", "overwegingen", and "beslissing"), \n'
                            '2 = Header Extraction: creates a dataframe with a column that holds a dictionary with '
                            'section header and section text, \n'
                            '3 = Section Extraction: creates a dataframe with a column that holds the section numbers '
                            'and section texts.'
                        ))
    dp_parser.add_argument('--input', type=str, default=constants.RAW_DIR.format(year=2021),
                           help="The path to the input data CSV file")
    
    # Segmentation pipeline arguments
    sp_parser = subparsers.add_parser('segmentation', help='Run the segmentation pipeline')
    sp_parser.add_argument('--method', type=int, choices=range(1, 6), default=1,
                           help=(
                               'Specify clustering method (1-5):\n'
                               '1 = TF-IDF + K-MEANS with seed words: clusters headers based on seed word groups,\n'
                               '2 = TF-IDF + K-MEANS with labeled data: clusters full text based on pre-labeled data,\n'
                               '3 = Self-Segmentation (Se3): clusters sections using the Se3 self-segmentation method,\n'
                               '4 = S-BERT + Spectral Clustering: clusters sections using S-BERT embeddings combined '
                               'with spectral clustering,\n'
                               '5 = LLM-based clustering: clusters sections using a Large Language Model-based approach.'
                           ))
    sp_parser.add_argument('--input', type=str, default=constants.SECTIONS_PATH.format(year=2020),
                           help="The path to the input data CSV file")
    sp_parser.add_argument('--eval', action='store_true', help="If true, returns the evaluation scores in a CSV file.")
    sp_parser.add_argument('--plot', action='store_true', help="If true, returns a plot of the clustered data (only "
                                                               "available for TFIDF + K-MEANS methods).")
    
    # Summarization pipeline arguments
    sm_parser = subparsers.add_parser('summarization', help='Run the summarization pipeline')
    sm_parser.add_argument('--method', type=int, choices=range(1, 5), default=1,
                           help=(
                               'Specify summarization method (1-4):\n'
                               '1 = TextRank (extractive summarization),\n'
                               '2 = BERT (extractive summarization),\n'
                               '3 = BART (abstractive summarization),\n'
                               '4 = Llama3.1-8B:instruct (abstractive summarization).'
                           ))
    sm_parser.add_argument('--input', type=str, default=constants.SECTIONS_PATH.format(year=2021),
                           help="The path to the input data CSV file")
    sm_parser.add_argument('--eval', action='store_true', help="If true, returns the evaluation scores in a CSV file.")
    
    args = parser.parse_args()
    
    if args.pipeline == 'data_processing':
        logger.info("Start data processing pipeline...")
        data_processor = DataProcessing()
        data_processor.data_process_selector(args.method, args.input)
        logger.info("Data processing pipeline successfully finished!")
    
    elif args.pipeline == 'segmentation':
        logger.info("Start segmentation pipeline...")
        segmentation_pipe = SegmentationPipeline()
        segmentation_pipe.segmentation_process_selector(args.method, args.input, args.eval, args.plot)
        logger.info("Segmentation pipeline successfully finished!")
    
    elif args.pipeline == 'summarization':
        logger.info("Start summarization pipeline...")
        summarization_pipe = SummarizationPipeline()
        summarization_pipe.summarization_process_selector(args.method, args.input, args.eval)
        logger.info("Summarization pipeline successfully finished!")
    
    else:
        parser.print_help()
