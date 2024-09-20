import os

REPO_PATH = ""

EXTRACTION_LOGGER_NAME = 'Extraction Pipeline'
SEGMENTATION_LOGGER_NAME = 'Segmentation Pipeline'
SUMMARIZATION_LOGGER_NAME = 'Summarization Pipeline'

#########################################
#     Paths, Columns and variables      #
#########################################

# FOLDER PATHS
DATA_DIR = os.path.join(REPO_PATH, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
RAW_DIR = os.path.join(DATA_DIR, 'raw', '{year}')
DATA_PROCESSING_DIR = os.path.join(REPO_PATH, 'data_processing')
SEGMENTATION_DIR = os.path.join(REPO_PATH, 'segmentation')
SEGMENTATION_RESULTS_DIR = os.path.join(SEGMENTATION_DIR, 'results')
MODEL_DIR = os.path.join(REPO_PATH, '', 'models')
SUMMARIZATION_DIR = os.path.join(REPO_PATH, 'summarization')
SUMMARIZATION_RESULTS_DIR = os.path.join(SUMMARIZATION_DIR, 'results')

# FILE PATHS
MERGE_PATTERNS_PATH = os.path.join(DATA_PROCESSING_DIR, 'data_processing/merge_patterns.txt')
SPLIT_PATTERNS_PATH = os.path.join(DATA_PROCESSING_DIR, 'data_processing/split_patterns.txt')
LABELED_HEADERS_FILE_PATH = os.path.join(METADATA_DIR, 'labeled_headers.csv')
HEADERS_PATH = os.path.join(METADATA_DIR, 'headers_{year}.csv')
SECTIONS_PATH = os.path.join(METADATA_DIR, 'sections_{year}.csv')
LLM_SYS_PROMPT_PATH = os.path.join(SEGMENTATION_DIR, 'system_prompt.txt')
LLM_PROMPT_NL_PATH = os.path.join(SEGMENTATION_DIR, 'prompt_nl.txt')
EMBEDDINGS_SAVE_PATH = os.path.join(SEGMENTATION_DIR)

# COLUMNS
ECLI_COL = 'ecli'
DATE_COL = 'date'
INSTANTIE_COL = 'legal_body'
INHOUD_COL = 'inhoudsindicatie'
OVERWEGINGEN_COL = 'overwegingen'
SECTIONS_COL = 'sections'
FULLTEXT_COL = 'fulltext'
TOKENIZED_COL = 'tokenized'
CLUSTER_COL = 'cluster'
LLM_RESPONSE_COL = 'llm_response'

# MODELS
DUTCH_BERT = 'textgain/allnli-GroNLP-bert-base-dutch-cased'
LEGAL_MULTILING_TF = 'joelito/legal-xlm-roberta-base'

#########################################
#       Segmentation Components         #
#########################################

SEED_WORDS_LIST = [
    # Procesverloop (procedure)
    [
        "procesverloop", "de procedure", "het geding in hoger beroep", "het procesverloop",
        "het geding in eerste aanleg", "ontstaan en loop van het geding", "de omvang van het geschil", "procesgang",
        "procesverloop in cassatie", "het verloop van de procedure", "procesverloop",
        "het verloop van het geding in eerste aanleg", "het verdere verloop van het geding in hoger beroep",
        "het verloop van het geding", "het oordeel van het hof", "het verloop van de procedure in hoger beroep",
        "feiten en procesverloop", "de procedure in hoger beroep", "procesverloop in hoger beroep", "procedure",
        "het verdere verloop van de procedure in hoger beroep", "het verdere verloop van de procedure",
        "de procedure bij de rechtbank", "de procedure in eerste aanleg", "verloop van de procedure",
        "het verdere procesverloop", "procedure bij de rechtbank", "het geschil en de beslissing in eerste aanleg"
    ],
    # Context
    [
        "context_overig", "voorvragen", "de voorvragen", "inleiding", "grondslag en inhoud van het eab",
        "onderzoek van de zaak", "het onderzoek ter terechtzitting", "onderzoek ter terechtzitting", "gronden",
        "het onderzoek op de terechtzitting", "onderzoek op de terechtzitting", "de kern van de zaak",
        "waar gaat deze zaak over?", "inleiding en samenvatting", "waar gaat het over?", "het bewijs",
        "waar gaat de zaak over?"
    ],
    # Feiten (facts)
    [
        "feiten", "de feiten", "feiten", "de zaak in het kort", "de vaststaande feiten", "vaststaande feiten",
        "uitgangspunten en feiten", "feiten en procesverloop", "feitelijke achtergrond", "uitgangspunten in cassatie"
    ],
    # Wettelijk kader (legal framework)
    [
        "wettelijk_kader", "toepasselijke wettelijke voorschriften", "de wettelijke voorschriften",
        "toepasselijke wetsbepalingen", "toepasselijke wetsartikelen", "de toegepaste wettelijke voorschriften",
        "de toepasselijke wetsartikelen", "de toegepaste wettelijke bepalingen"
    ],
    # Beoordeling (assessment)
    [
        "beoordeling_door_rechter", "de beoordeling", "waardering van het bewijs", "de motivering van de beslissing",
        "beoordeling", "bewezenverklaring", "de beoordeling van het bewijs", "de verdere beoordeling",
        "beoordeling van het geschil", "de strafbaarheid van de verdachte", "strafbaarheid van verdachte",
        "de strafbaarheid van verdachte", "de strafbaarheid", "beoordeling door de rechtbank", "strafbaarheid",
        "beoordeling van het middel", "strafbaarheid van de feiten", "de strafbaarheid van het bewezenverklaarde",
        "conclusie en gevolgen", "de bewezenverklaring", "de strafbaarheid van de feiten",
        "de strafbaarheid van het bewezen verklaarde", "beoordeling van het cassatiemiddel",
        "beoordeling van de middelen",
        "bespreking van het cassatiemiddel", "de bewijsmotivering", "beoordeling van het bewijs",
        "strafbaarheid verdachte", "de motivering van de beslissing in hoger beroep",
        "beoordeling van de ontvankelijkheid van het beroep in cassatie", "de bewijsbeslissing",
        "de kwalificatie van het bewezenverklaarde", "strafbaarheid van de verdachte", "de gronden van de beslissing",
        "motivering van de straf", "de strafbaarheid van het feit", "de beoordeling van het geschil",
        "strafbaarheid van het feit", "motivering straf", "motivering van de sanctie", "het oordeel van de rechtbank",
        "de beoordeling in het incident", "beoordeling van de klachten", "de beoordeling in hoger beroep",
        "overwegingen",
        "kwalificatie en strafbaarheid van de feiten", "beoordeling van de cassatiemiddelen",
        "beoordeling van het eerste cassatiemiddel", "strafbaarheid feiten",
        "de overwegingen ten aanzien van straf en/of maatregel", "beoordeling van het tweede cassatiemiddel",
        "beoordeling in hoger beroep", "strafbaarheid feit", "motivering van de straffen en maatregelen",
        "kwalificatie en strafbaarheid van het feit", "de beoordeling in conventie en in reconventie",
        "het geschil in reconventie", "de straf en/of de maatregel", "bewijs", "het eerste middel",
        "de beoordeling van de civiele vordering"
    ],
    # Beslissing (conclusion)
    [
        "beslissing", "de beslissing", "beslissing", "slotsom", "de uitspraak", "de slotsom", "de strafoplegging",
        "conclusie", "oplegging van straf", "de op te leggen straf of maatregel", "vrijspraak", ".beslissing",
        ". beslissing", "de straf"
    ],
    # Proceskosten (legal costs)
    ["proceskosten", "proceskosten", "griffierecht en proceskosten", "kosten", "proceskosten en griffierecht"],
    # Proceshandelingen partijen (procedural acts of the parties)
    [
        "proceshandelingen_partijen", "het geschil", "tenlastelegging", "de tenlastelegging", "de vordering",
        "het verweer", "het verzoek", "geschil", "geding in cassatie", "eis officier van justitie",
        "de vordering en het verweer", "de inhoud van de tenlastelegging", "het wrakingsverzoek",
        "geschil in hoger beroep", "het verzoek en het verweer", "geschil en conclusies van partijen",
        "het cassatieberoep", "het geschil in conventie", "de standpunten", "standpunten van partijen",
        "het standpunt van de officier van justitie", "de vordering tot tenuitvoerlegging"
    ]
]

ADDITIONAL_DUTCH_STOPWORDS = ['de', 'het', 'een', 'en', 'van', 'in', 'op', 'aan', 'met', 'voor', 'over', 'onder',
                              'tussen', 'door', 'uit', 'naar', 'bij']

LEGAL_TOPIC_LIST = ['feiten en omstandigheden',
                    'eerdere juridische acties en beslissingen (tevens onderscheid tussen gedaagde, appellant en de '
                    'verschillende juridische instanties zoals rechtbank, hof van beroep etc.)',
                    'standpunten van appellant',
                    'standpunten van verweerder',
                    'juridische middelen',
                    'beoordeling door rechter/College',
                    'proceskosten']
