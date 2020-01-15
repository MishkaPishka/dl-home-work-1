import dynet as dy




class GENERAL_MODEL(object):
    def __init__(self, model, repr_function, input_word_size, lstm1_dim, lstm2_dim, hid_dim, out_dim,voc_sizes,dim_sizes):

        lookup_parameters_list = model.lookup_parameters_list()
        for emb in lookup_parameters_list:
            if emb.name()=='/WordEmbeds':
                self.word_embeds =emb
                self.H0 = model.add_parameters((input_word_size, 2 * dim_sizes['word_representation']))
            elif emb.name()=='/CharEmbeds':
                self.char_embeds =emb
                self.cFwdRNN = dy.LSTMBuilder(1, dim_sizes['char_representation'], dim_sizes['word_representation'],
                                              model)
            elif emb.name()=='/SuffixEmbeds':
                self.suffix_embeds = emb
            elif emb.name()=='/PrefixEmbeds':
                self.prefix_embeds = emb



        self.fwdRNN = dy.LSTMBuilder(1, input_word_size, lstm1_dim, model)
        self.bwdRNN = dy.LSTMBuilder(1, input_word_size, lstm1_dim, model)

        self.fwdRNN_2 =  dy.LSTMBuilder(1, lstm1_dim * 2, lstm2_dim, model)
        self.bwdRNN_2 =  dy.LSTMBuilder(1, lstm1_dim * 2, lstm2_dim, model)

        self.pH = model.add_parameters((hid_dim, lstm2_dim * 2))
        self.pO = model.add_parameters((out_dim, hid_dim))
        self.repr_function = repr_function

    def __call__(self,sequence):
        dy.renew_cg()

        wembs = [self.repr_function(self,word) for word in sequence]

        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))
        bi = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
        f_init_2 = self.fwdRNN_2.initial_state()
        b_init_2 = self.bwdRNN_2.initial_state()
        fw_exps_2 = f_init_2.transduce(bi)
        bw_exps_2 = b_init_2.transduce(reversed(bi))
        bi_tag = [dy.concatenate([f, b]) for f, b in zip(fw_exps_2, reversed(bw_exps_2))]
        H = self.pH.expr()
        O = self.pO.expr()
        outs = [O * (dy.tanh(H * x)) for x in bi_tag]
        return outs
    #
    # def get_rpr_d(self,sequence):
    #     char_embs = [self.char_embeds[cid] for cid in sequence[1]]
    #     fw_exps = self.cFwdRNN.initial_state()
    #     ans= fw_exps.transduce(char_embs)
    #     return dy.concatenate([self.word_embeds[sequence[0]],ans[-1]])

    def get_rpr_d(self,sequence):
        char_embs = [self.char_embeds[cid] for cid in sequence[1]]
        fw_exps = self.cFwdRNN.initial_state()
        ans= fw_exps.transduce(char_embs)
        ans = dy.concatenate([self.word_embeds[sequence[0]],ans[-1]])
        H0 = self.H0.expr()
        return   dy.tanh(H0 * ans)

    def get_rpr_b(self,sequence):
        char_embs = [self.char_embeds[cid] for cid in sequence]
        fw_exps = self.cFwdRNN.initial_state()
        ans= fw_exps.transduce(char_embs)

        return ans[-1]

    def get_rpr_a(self,w):
        return self.word_embeds[w]

    def get_rpr_c(self, w):
        return dy.esum([self.word_embeds[w[0][0]],self.prefix_embeds[w[0][1]],self.suffix_embeds[w[0][2]]])