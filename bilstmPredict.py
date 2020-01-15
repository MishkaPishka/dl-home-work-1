import argparse
import pickle
import random
import time
from pathlib import Path

import numpy
import torch

import parse_code

import dynet as dy

# TODO
# set number to 5
from bilstmTrain import  create_model




if __name__ == '__main__':

    tagger_mode, repr_mode, model_file, inputFile = parse_code.get_command_line_arguments_predict()

    print("Variables:\ntagger_mode{}\n,repr_mode:{}\n,model_file:{}\n".format(tagger_mode, repr_mode, model_file))


    words_dictinary, charachters_dictinary, labels_dictinary, prefix_dictinary, suffix_dictinary, dim_sizes, voc_sizes = pickle.load(
        open(model_file + ".data", 'rb'))
    rev_labels = {i: k for k, i in labels_dictinary.items()}
    model, net = create_model(repr_mode, dim_sizes, voc_sizes)
    model.populate(model_file)

    sentances = parse_code.create_sentences_test(inputFile)

    output = 'test_output_repr_' + repr_mode + "." + tagger_mode
    dictionary_list = [words_dictinary, charachters_dictinary, labels_dictinary, prefix_dictinary, suffix_dictinary]
    parse_code.dictionary_list = dictionary_list
    with open(output, 'w') as f:

        for i, sentance in enumerate(sentances):
            dy.renew_cg()
            # vecs, labels = parse_sequence(sequence)
            # vecs, labels = parse_sequence_to_chars(sequence)
            lower_sentance = [str.lower(s) for s in sentance]
            vecs = parse_code.parse_sequence_to_vectors(lower_sentance, repr_mode,test_mdee=True)[0]

            # vecs = predictor.repr_vecs(vecs) #embed / ...
            # print(vecs)
            # print(labels)
            preds = net(vecs)

            for i in range(0, len(preds)):
                preds[i] = dy.log_softmax(preds[i])

            probs = [v.npvalue() for v in preds]
            tags = []
            for k, prb in enumerate(probs):
                tag = numpy.argmax(prb)
                tags.append(tag)
                tag_name = str.upper(rev_labels[tag])
                f.write(sentance[k] + " " + tag_name + "\n")

            f.write('' + '\n')
        f.close()
    exit(0)

