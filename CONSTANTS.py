from pathlib import Path

PREFIX_SUFFIX_LENGTH = 3
UNKNOWN_WORD_VAL = '__UUNNKK__'
TAGGER_MODE_NER = "ner"
TAGGER_MODE_POS = "pos"

REPR_MODE_A = 'a'
REPR_MODE_B = 'b'
REPR_MODE_C = 'c'
REPR_MODE_D = 'd'

EMBEDDING_WORD_SIZE = 50
EMBEDING_CHAR_SIZE = 50
CHR_LSTM_OUTPUT = EMBEDDING_WORD_SIZE
BATCH_SIZE = 8

default_data_path = Path(__file__).parent / 'data'

default_external_word_vectors_path = default_data_path / "wordVectors.txt"
default_external_word_vocab_path = default_data_path / "vocab.txt"
DEFAULT_TAGGER = TAGGER_MODE_NER

DEFAULT_REPR_MODE = REPR_MODE_A

TAGGER_MODE = None
REPR_MODE = None
MODAL_FILE = None
def get_file_path(file_name):
    return default_data_path/ TAGGER_MODE / file_name



