# from allennlp.modules.elmo import Elmo, batch_to_ids
#
# options_file = "/Users/feili/Downloads/elmo_2x1024_128_2048cnn_1xhighway_options.json"
# weight_file = "/Users/feili/Downloads/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
#
# elmo = Elmo(options_file, weight_file, 2, dropout=0)
#
# # use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', '.'], ['Another', '.']]
# character_ids = batch_to_ids(sentences)
#
# embeddings = elmo(character_ids)


from elmoformanylangs import Embedder

def test_elmoformanylangs():

    e = Embedder('/Users/feili/project/ELMoForManyLangs/output/en')
    # e = Embedder('/Users/feili/resource/data_to_train_emb/elmo_ade_lower_0norm_200d')

    sents = [['LABA', ',', 'such', 'as', 'vilanterol']]
    # for idx, sent in enumerate(sents):
    #     for idy, tk in enumerate(sent):
    #         sents[idx][idy] = sents[idx][idy].lower()

    ret = e.sents2elmo(sents)
    # will return a list of numpy arrays
    # each with the shape=(seq_len, embedding_size)

    pass




if __name__ == '__main__':
    test_elmoformanylangs()
    pass
