## Data
For this project data from Rechtspraak Open Data is used.

### Data directory setup
The data from Rechtspraak Open Data contain folders for each year with XML files that contain the information in the documents.
To ensure the scripts work on the data, create a folder called data and store the files from the downloaded Rechtspraak Open data zip in the following way:

```commandline
data/
├── creation/
├── metadata/
└── raw/
    ├── 2020/
    │   └── xml files...
    ├── 2021/
    │   └── xml files...
    └── etc...
```

## Data Processing Script for Rechtspraak Case Documents

This script is designed to process legal documents from the Rechtspraak dataset. It extracts useful information such as full text, specific sections, or headers from the documents and saves the processed data into a CSV file. The script leverages different extraction techniques based on the chosen processing method.

### Features

- **Full Text Extraction**: Extracts the entire content of the document, including "procesverloop", "overwegingen", and "beslissing".
- **Header Extraction**: Extracts headers along with their corresponding sections, providing additional structured data.

### Dependencies

The script uses the following Python libraries:
- `pandas`: For data manipulation and creating the CSV output.
- `argparse`: For handling command-line arguments.
- Custom modules such as `header_extraction` and `full_text_extraction`.

### Usage

You can run the script from the command line with different options depending on what kind of data you want to extract from the documents.

#### Command-Line Options

- `--method` (required): Specifies the type of processing method to use.
  - `1`: **Full Text Extraction**: Extracts the full text of the document.
  - `2`: **Header Extraction**: Extracts headers and their corresponding sections.

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
- **Sentence-based Clustering**:
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

