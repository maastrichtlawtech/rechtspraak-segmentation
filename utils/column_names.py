from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnNames:
    ecli = 'ecli'
    date = 'date'
    instantie = 'legal_body'
    inhoudsindicatie = 'inhoudsindicatie'
    overwegingen = 'overwegingen'
    sections = 'sections'
    fulltext = 'fulltext'
    tokenized = 'tokenized'
    cluster = 'cluster'
    llm_response = 'llm_response'

COLS = ColumnNames()