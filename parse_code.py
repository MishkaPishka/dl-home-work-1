import argparse

from torch.utils.data import Dataset

import CONSTANTS

unknown_word_repr = "<UNK>"

PADDING_WORD_RPR = "<PAD>"
PADDING_WORD_TAG = "<PAD>"
PREFIX_SUFFIX_LENGTH = 3
ENCCODING_TYPE = 'IDX'

dictionary_list = []



def create_sentences_test( data_file_path):
    sentences = []
    with open(data_file_path) as f:
        current_sentence = []
        for idx, line in enumerate(f):
            if line == "\n":
                sentences.append(current_sentence)
                current_sentence = []
                continue

            item = line.strip()
            current_sentence.append(item)

    return sentences


def load_sentences( data_file_path):
    sentences = []
    words = set()
    charachters =  set()
    labels =  set()
    prefix = set()
    suffix = set ()
    with open(data_file_path) as file:
        current_sentence = []
        for idx, line in enumerate(file):
            if line == '\n':
                sentences.append(current_sentence)
                current_sentence = []
                continue
            line_items = line.split()
            line_items = [word.lower() for word in line_items]
            if len(line_items)>1:
                words.add(line_items[0])
                labels.add(line_items[1])
                charachters.update([i for i in line_items[0]])
                prefix.add(line_items[0][:3])
                suffix.add(line_items[0][-3:])
                # print(line_items[0])
                # print(line_items[0][-3:])

            current_sentence.append(line_items)

    return sentences,words,labels,charachters,prefix,suffix


def set_to_ditionary_and_reversed(some_set,labels=False):
    dictionary = {i: j for j,i in enumerate(some_set)}
    if labels != True:
        dictionary[CONSTANTS.UNKNOWN_WORD_VAL] = len(dictionary)
    rev_ditionary = {j: i for i,j in dictionary.items()}
    return dictionary,rev_ditionary


def category_to_arr(i):
    arr = [0,0]
    arr[i] = 1
    return arr

#REPRESENTATION 1 : ONE -HOT ab = [0.....1...] [0.....1...]
def char_to_idx(char,CHARS_TO_IDX):
    return CHARS_TO_IDX[char]
def idx_to_char(idx,IDX_TO_CHARS):
    return IDX_TO_CHARS[idx]

#return symbol representation as one-hot
def symbol_one_hot_encode(simbol,simbol_idx_map):
    features = [0]*len(simbol_idx_map)
    features[simbol_idx_map[simbol]] = 1
    return features

#return sequence of symbol representation as one-hot
def sequence_one_hot_encode(sequence,simbol_idx_map):
    arr = [[0] * len(simbol_idx_map) for i in range(len(sequence))]
    for i,j in enumerate(sequence):
        arr[i] = symbol_one_hot_encode(j)
    return arr

#REPRESENTATION 2: list of inxes ab = [10,11]

def symbol_idx_encoding(simbol,simbol_idx_map):
    return [simbol_idx_map[simbol]]
def sequence_idx_encoding(sequence,simbol_idx_map):
    arr =[0] * len(sequence)
    for i,j in enumerate(sequence):
        arr[i] = simbol_idx_map[j]
    return arr



class SEQUENCE_DATA(Dataset):

    def __init__(self, sentances,transform=None):
        self.data = sentances
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        example, lable =  [self.data[idx][x][0] for x in range(0, len(self.data[idx]))],[self.data[idx][x][1] for x in range(0, len(self.data[idx]))]

        if self.transform:
            example = self.transform(example)

        return example,lable

transform_method = sequence_one_hot_encode if ENCCODING_TYPE=='HOT' else sequence_idx_encoding


# returns sequences as wector of dictionary items
def parse_sequence_to_vectors(sequence, repr_mode, test_mdee=False):
    global dictionary_list

    [words_dictinary, charachters_dictinary, labels_dictinary, prefix_dictinary, suffix_dictinary] =dictionary_list
    # a words , #b chars #c word parts #d words+chars
    words = []
    labels = []
    only_chars = []

    d_input = []
    c_input = []
    word_sequence = sequence
    if not test_mdee:
     word_sequence = sequence[0]
     labels_sequence = sequence[1]
    for word in range(0, len(word_sequence)):
        if word_sequence[word] in words_dictinary.keys():
            w = word_sequence[word]
        else:
            w = CONSTANTS.UNKNOWN_WORD_VAL

        words.append((words_dictinary[w]))
        pref = prefix_dictinary[w[:3]] if w[:3] in prefix_dictinary.keys() else prefix_dictinary[CONSTANTS.UNKNOWN_WORD_VAL]
        suffix = suffix_dictinary[w[-3:]] if w[-3:] in suffix_dictinary.keys() else suffix_dictinary[CONSTANTS.UNKNOWN_WORD_VAL]

        c_input.append([(words_dictinary[w], pref, suffix)])
        chars = []

        for chr in word_sequence[word]:
            if chr in charachters_dictinary.keys():
                chars.append(charachters_dictinary[chr])
            else:
                chars.append(charachters_dictinary[CONSTANTS.UNKNOWN_WORD_VAL])
        only_chars.append(chars)
        d_input.append([words_dictinary[w], chars])
        # print( sequence[0][i])
        if test_mdee != True:
            labels.append(labels_dictinary[labels_sequence[word]])

    if repr_mode == CONSTANTS.REPR_MODE_A:
        input = words

    elif repr_mode == CONSTANTS.REPR_MODE_B:
        input = only_chars
    elif repr_mode == CONSTANTS.REPR_MODE_C:
        input = c_input
    else:
        input = d_input

    return input, labels

def get_command_line_arguments():
    ###
    #   PARSE VARIABLES
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagger-mode", "-tm", help="Set tagger mode: POS")
    parser.add_argument("--repr-mode", "-rm", type=bool, help="Set to any of: a, c")
    parser.add_argument("--model-file", "-mf", help="Path to output model file to.")
    args = parser.parse_args()
    tagger_mode = args.tagger_mode if args.tagger_mode is not None else CONSTANTS.DEFAULT_TAGGER
    repr_mode = args.repr_mode if args.repr_mode is not None else CONSTANTS.DEFAULT_REPR_MODE
    model_file = args.model_file if args.model_file is not None else "model" + repr_mode + tagger_mode

    CONSTANTS.TAGGER_MODE = tagger_mode
    CONSTANTS.REPR_MODE = repr_mode
    CONSTANTS.MODAL_FILE = model_file

    return tagger_mode,repr_mode,model_file

def get_command_line_arguments_predict():
    ###
    #   PARSE VARIABLES
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagger-mode", "-tm", help="Set tagger mode: POS")
    parser.add_argument("--repr-mode", "-rm", type=bool, help="Set to any of: a, c")
    parser.add_argument("--model-file", "-mf", help="Path to output model file to.")
    parser.add_argument("--test-file", "-tf", help="Path to output model file to.")


    args = parser.parse_args()
    tagger_mode = args.tagger_mode if args.tagger_mode is not None else  CONSTANTS.DEFAULT_TAGGER
    repr_mode = args.repr_mode if args.repr_mode is not None else  CONSTANTS.DEFAULT_REPR_MODE
    model_file = args.model_file if args.model_file is not None else  "model" + repr_mode + tagger_mode

    CONSTANTS.TAGGER_MODE = tagger_mode
    CONSTANTS.REPR_MODE = repr_mode
    CONSTANTS.MODAL_FILE = model_file

    test_file = args.model_file if args.test_file is not None else   CONSTANTS.get_file_path('test')

    return tagger_mode,repr_mode,model_file,test_file
