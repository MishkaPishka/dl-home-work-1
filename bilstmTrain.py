import pickle
import time

import numpy
import torch
import CONSTANTS
import dy_bilstm_models
import parse_code

import dynet as dy
import dynet_config

# dynet_config.set(autobatch=0)
# File paths
from dy_bilstm_models import GENERAL_MODEL

DEFAULT_NUMBER_OF_EPOCHS = 5


###
#
###
# training code
def collate_fn(data):
    return (data)


def create_model(repr_mode, dim_sizes, voc_sizes):
    input_word_size = dim_sizes['in_dim']
    lstm1_dim = dim_sizes['lstm1_dim']
    lstm2_dim = dim_sizes['lstim2_dim']
    hid_dim = dim_sizes['hid_dim']
    out_dim = dim_sizes['out_dim']
    model = dy.Model()

    if repr_mode == CONSTANTS.REPR_MODE_A:
        words_lookup = model.add_lookup_parameters((voc_sizes['words'] + 1, dim_sizes['word_representation']),
                                                   name='WordEmbeds')
        repr_function = dy_bilstm_models.GENERAL_MODEL.get_rpr_a

    if repr_mode == CONSTANTS.REPR_MODE_B:
        model.add_lookup_parameters((voc_sizes['chars'] + 1, dim_sizes['char_representation']), name='CharEmbeds')
        repr_function = dy_bilstm_models.GENERAL_MODEL.get_rpr_b

    if repr_mode == CONSTANTS.REPR_MODE_C:
        words_lookup = model.add_lookup_parameters((voc_sizes['words'] + 1, dim_sizes['word_representation']),
                                                   name='WordEmbeds')
        suffix_lookup = model.add_lookup_parameters((voc_sizes['suffix'] + 1, dim_sizes['suffix_representation']),
                                                    name='SuffixEmbeds')
        prefix_lookup = model.add_lookup_parameters((voc_sizes['prefix'] + 1, dim_sizes['prefix_representation']),
                                                    name='PrefixEmbeds')
        repr_function = dy_bilstm_models.GENERAL_MODEL.get_rpr_c

    if repr_mode == CONSTANTS.REPR_MODE_D:
        words_lookup = model.add_lookup_parameters((voc_sizes['words'] + 1, dim_sizes['word_representation']),
                                                   name='WordEmbeds')
        model.add_lookup_parameters((voc_sizes['chars'] + 1, dim_sizes['char_representation']), name='CharEmbeds')
        repr_function = dy_bilstm_models.GENERAL_MODEL.get_rpr_d
        dim_sizes['in_dim'] = 2 * dim_sizes['word_representation']
        input_word_size = dim_sizes['in_dim']
    # insert embeding layer

    return model, GENERAL_MODEL(model, repr_function, input_word_size, lstm1_dim, lstm2_dim, hid_dim, out_dim,
                                voc_sizes, dim_sizes)


def train(train_data, dev_data, model, predictor, repr_mode, trainer, loss1):
    train_losses = []
    dev_losses = []
    dev_accuracies = []

    ctr = 0
    epoch_start_time = time.time()
    for epoch in range(DEFAULT_NUMBER_OF_EPOCHS):
        # Iterate over sentance to get batch_Size number of examples

        epoch_loss = 0
        # TRAIN
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=CONSTANTS.BATCH_SIZE,
                                                 shuffle=True,
                                                 collate_fn=collate_fn)
        for sequences in dataloader:
             for sequence in sequences:
                ctr += 1

                vecs, labels = parse_code.parse_sequence_to_vectors(sequence, repr_mode)
                preds =  dy.softmax(dy.concatenate_to_batch(predictor(vecs)))
                loss_sym = dy.sum_batches(loss(preds, labels))
                loss_sym.forward()
                loss_sym.backward()
                trainer.update()
                dy.renew_cg()



        # for sequences in dataloader:
        #     # OPTION 1 : BATCH SIZE * SEUQENCE BATCHES
        #     diff_preds = []
        #     dif_lables = []
        #     for sequence in sequences:
        #         ctr += 1
        #         vecs, labels = parse_code.parse_sequence_to_vectors(sequence, repr_mode)
        #         for v in vecs:
        #             diff_preds.append(v)
        #         for l in labels:
        #             dif_lables.append(l)
        #
        #     a = predictor(diff_preds)
        #
        #     preds = dy.concatenate_to_batch(a)
        #     # preds = dy.log_softmax(preds)
        #     loss_sym = dy.sum_batches(dy.pickneglogsoftmax_batch(preds, dif_lables))
        #     loss_sym.forward()
        #     loss_sym.backward()
        #     trainer.update()

                if ctr % 5000 == 0:

                    total = 0.000000
                    acc = 0.0
                    current_dev_loss = 0.0
                    o_tag = labels_dictinary['o'] if 'o' in labels_dictinary.keys() else -1
                    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=CONSTANTS.BATCH_SIZE,
                                                                 shuffle=True, collate_fn=collate_fn)
                    for sequences in dev_dataloader:

                        # parse to sequence
                        for sequence in sequences:
                            losses = []

                            dy.renew_cg()
                            vecs, labels = parse_code.parse_sequence_to_vectors(sequence, repr_mode)
                            preds = predictor(vecs)

                            for i in range(0, len(preds)):
                                #  preds[i] = dy.log_softmax(preds[i])
                                x = dy.pickneglogsoftmax(preds[i], labels[i])
                                preds[i] = dy.log_softmax(preds[i])

                                current_dev_loss += x.scalar_value()

                            probs = [v.npvalue() for v in preds]
                            tags = []
                            for prb in probs:
                                tag = numpy.argmax(prb)
                                tags.append(tag)
                            total += len([i for i in range(0, len(labels)) if
                                          (labels[i] != tags[i] or (tags[i] == labels[i] and labels[i] != o_tag))])
                            acc += len([i for i in range(0, len(labels)) if (tags[i] == labels[i] and labels[i] != o_tag)])

                    print(acc/total)
                    dev_losses.append(current_dev_loss)
                    dev_accuracies.append(acc / total)


    print("Done training.")
    print(dev_accuracies)
    return dev_losses, dev_accuracies




if __name__ == '__main__':
    tagger_mode, repr_mode, model_file = parse_code.get_command_line_arguments()

    ##
    #   PRINT VARIABLES
    ##
    print("Variables:\ntagger_mode{}\n,repr_mode:{}\n,model_file:{}\n".format(tagger_mode, repr_mode, model_file))

    ###
    # PARS VOCABULARY
    ###
    file_path = CONSTANTS.get_file_path( 'train')

    sentences, words, labels, charachters, prefix, suffix = parse_code.load_sentences(
        file_path)  # sentance = a sequence


    words_dictinary, words_rev_dictionary = parse_code.set_to_ditionary_and_reversed(words)
    labels_dictinary, labels_rev_dictionary = parse_code.set_to_ditionary_and_reversed(labels, labels=True)

    charachters_dictinary, charachters_rev_dictionary = parse_code.set_to_ditionary_and_reversed(charachters)
    prefix_dictinary, prefix_rev_dictionary = parse_code.set_to_ditionary_and_reversed(prefix)
    suffix_dictinary, suffix_rev_dictionary = parse_code.set_to_ditionary_and_reversed(suffix)
    dictionary_list = [words_dictinary, charachters_dictinary, labels_dictinary, prefix_dictinary, suffix_dictinary]
    parse_code.dictionary_list = dictionary_list
    voc_size = len(words)
    labels_size = len(labels)
    char_size = len(charachters)
    num_sentances = len(sentences)
    voc_sizes = {}
    voc_sizes['words'] = voc_size
    voc_sizes['chars'] = char_size
    voc_sizes['prefix'] = len(prefix_dictinary)
    voc_sizes['suffix'] = len(suffix_dictinary)
    out_dim = labels_size

    in_dim = CONSTANTS.EMBEDDING_WORD_SIZE if repr_mode == CONSTANTS.REPR_MODE_A or repr_mode == CONSTANTS.REPR_MODE_B else 2 * CONSTANTS.EMBEDDING_WORD_SIZE

    ###
    #   LOAD EXAMPLES and TESTS
    ###
    train_data = parse_code.SEQUENCE_DATA(sentences)
    dev_path = CONSTANTS.get_file_path('dev')
    dev_data = parse_code.SEQUENCE_DATA(parse_code.load_sentences(dev_path)[0])  # sentance = a sequence

    ###
    # DEFINE MODEL
    ###

    lstm1_dim = 30
    lstim2_dim = 20
    out_dim = len(labels)
    hid_dim = 30

    dim_sizes = {}

    dim_sizes['out_dim'] = out_dim
    dim_sizes['lstm1_dim'] = lstm1_dim
    dim_sizes['lstim2_dim'] = lstim2_dim
    dim_sizes['hid_dim'] = hid_dim
    dim_sizes['char_representation'] = CONSTANTS.EMBEDING_CHAR_SIZE
    dim_sizes['word_representation'] = CONSTANTS.EMBEDDING_WORD_SIZE
    in_dim = CONSTANTS.EMBEDDING_WORD_SIZE
    dim_sizes['in_dim'] = in_dim

    dim_sizes['suffix_representation'] = dim_sizes['word_representation']
    dim_sizes['prefix_representation'] = dim_sizes['word_representation']

    model, net = create_model(repr_mode, dim_sizes, voc_sizes)

    trainer = dy.AdamTrainer(model)
    loss = dy.pickneglogsoftmax_batch

    # TRAIN model
    dev_losses, dev_accuracies = train(train_data, dev_data, model, net, repr_mode, trainer, loss)

    with open(f'pos_{repr_mode}_visual_stats.pkl', 'wb') as f:
        pickle.dump({'dev_accuracies': dev_accuracies, 'dev_losses': dev_losses}, f)



    # SAVE MODEL

    pickle.dump(
        [words_dictinary, charachters_dictinary, labels_dictinary, prefix_dictinary, suffix_dictinary, dim_sizes,
         voc_sizes], open(model_file + ".data", 'wb'))
    model.save(model_file)
