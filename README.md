# Master Thesis
## Optimizing Dutch Legal Text Segmentation and Summarization: _A Low-Resource Approach to Enhancing Relevance Extraction in Complex Case Law_
**Abstract**: _This thesis addresses the complexities inherent in analyzing Dutch legal case law, which is characterized by its voluminous and intricate documentation. The research focuses on developing effective methods for segmenting these legal texts, with a particular emphasis on identifying and extracting relevant information. In the legal context, relevance extends beyond simple topical or lexical similarities, encompassing the intricate legal relationships between entities that form the basis of judicial decisions.
The study explores various approaches to divide legal documents into coherent sections based on these relevant legal topics. By combining clustering techniques for legal topics and investigating summarization methods for both full documents and segmented documents, this research aims to determine the most efficient strategies for generating accurate and concise summaries.
A key consideration in this work is the constraint of limited resources, including computational power and high-quality reference summaries. The thesis addresses these challenges, seeking solutions that are effective under such constraints.
To validate the proposed methodologies, the research employs a comprehensive evaluation framework, incorporating both automated metrics and human expert assessment. The ultimate goal of this work is to enhance the capacity of legal professionals to efficiently extract crucial information from complex legal documents, thereby streamlining their analysis and decision-making processes._

**Keywords**: _Dutch legal case law, legal text segmentation, information extraction, low-resource, relevance in legal context, human evaluation_ 

---
## Data Processing Script for Rechtspraak Case Documents

This script is designed to process legal documents from the Rechtspraak dataset. It extracts useful information such as full text, specific sections, or headers from the documents and saves the processed data into a CSV file. The script leverages different extraction techniques based on the chosen processing method.

### Features

- **Full Text Extraction**: Extracts the entire content of the document, including "procesverloop", "overwegingen", and "beslissing".
- **Section Extraction**: Extracts and organizes the main sections of the document into separate columns.
- **Header Extraction**: Extracts headers along with their corresponding sections, providing additional structured data.

### Dependencies

The script uses the following Python libraries:
- `pandas`: For data manipulation and creating the CSV output.
- `argparse`: For handling command-line arguments.
- Custom modules such as `header_extraction`, `full_text_extraction`, and `section_extraction`.

### Usage

You can run the script from the command line with different options depending on what kind of data you want to extract from the documents.

#### Command-Line Options

- `--method` (required): Specifies the type of processing method to use.
  - `1`: **Full Text Extraction**: Extracts the full text of the document.
  - `2`: **Main Section Extraction**: Extracts and organizes the main sections into separate columns.
  - `3`: **Header Extraction**: Extracts headers and their corresponding sections.

- `--input` (required): Specifies the path to the input CSV file containing the document data.

#### Example Commands

- **Full Text Extraction**:
  ```bash
  python summarization_pipeline.py --method 1 --input path/to/input_file.csv
  ```
- **Header Extraction**:
  ```bash
  python summarization_pipeline.py --method 2 --input path/to/input_file.csv
  ```
  
### Output
The script processes the data according to the selected method and saves the results into a CSV file at a predefined location. The file name and save path can be adjusted in the script as needed.

### Logging
The script uses a logger to provide detailed information about the processing steps and any issues encountered during execution. Logs are generated for each run, making it easier to trace the execution flow and diagnose problems.

---
## Segmentation Pipeline for Rechtspraak Case Documents

This script is designed to cluster legal document data from the Rechtspraak dataset. It categorizes document headers, full texts, or specific sections using various clustering techniques such as TF-IDF with K-Means, self-segmentation, and spectral clustering. The results are saved as CSV files for further analysis.

### Features

- **Header Clustering**: Clusters headers based on seed word groups using TF-IDF and K-Means.
- **Full Text Clustering**: Clusters the full text of documents using TF-IDF and K-Means based on pre-labeled data.
- **Section Clustering**:
  - **Self-Segmentation (Se3)**: Clusters sections using the Se3 self-segmentation method.
  - **S-BERT + Spectral Clustering**: Clusters sections using S-BERT embeddings combined with spectral clustering.
- **LLM-based Clustering**: Clusters sections using a Large Language Model-based approach.

### Dependencies

The script uses the following Python libraries:
- `pandas`: For data manipulation and creating the CSV output.
- `argparse`: For handling command-line arguments.
- Custom modules such as `tfidf_kmeans`, `se3`, `sbert_spectral`, and `llm`.

### Usage

You can run the script from the command line with different options depending on the clustering method you want to apply to the documents.

#### Command-Line Options

- `--method` (required): Specifies the type of clustering method to use.
  - `1`: **TF-IDF + K-MEANS with seed words**: Clusters headers based on predefined seed word groups.
  - `2`: **TF-IDF + K-MEANS with labeled data**: Clusters full text using pre-labeled data for more accurate categorization.
  - `3`: **Self-Segmentation (Se3)**: Clusters document sections using a self-segmentation approach.
  - `4`: **S-BERT + Spectral Clustering**: Clusters document sections using S-BERT embeddings combined with spectral clustering.
  - `5`: **LLM-based Clustering**: Clusters sections using a Large Language Model-based approach.

- `--input` (required): Specifies the path to the input CSV file containing the document data.

#### Example Commands

- **TF-IDF + K-MEANS with seed words**:
  ```bash
  python segmentation_pipeline.py --method 1 --input path/to/input_file.csv
  ```
- **TF-IDF + K-MEANS with labeled data**:
  ```bash
  python segmentation_pipeline.py --method 2 --input path/to/input_file.csv
  ```
  - **Self-Segmentation (Se3)**:
  ```bash
  python segmentation_pipeline.py --method 3 --input path/to/input_file.csv
  ```
  - **S-BERT + Spectral Clustering**:
  ```bash
  python segmentation_pipeline.py --method 4 --input path/to/input_file.csv
  ```
  - **LLM-based Clustering**:
  ```bash
  python segmentation_pipeline.py --method 5 --input path/to/input_file.csv
  ```

### Output
The script clusters the data according to the selected method and saves the results into a CSV file at a predefined location. The file name and save path can be adjusted within the script.

### Logging
The script uses a logger to provide detailed information about the clustering steps and any issues encountered during execution. Logs are generated for each run, making it easier to trace the execution flow and diagnose problems.


### Summary:

- **Header Clustering**: Describes how headers are clustered using TF-IDF and K-Means.
- **Full Text Clustering**: Explains full-text clustering based on labeled data.
- **Section Clustering**: Outlines the various section clustering methods, including self-segmentation and spectral clustering.
- **LLM-based Clustering**: Mentions the approach using a Large Language Model for clustering.
- **Usage and Commands**: Provides example command-line usage and explains the purpose of each method.

