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


    sents = [['Labour', 'Party', 'yes'],
    ['Legacy']]

    ret = e.sents2elmo(sents)
    # will return a list of numpy arrays
    # each with the shape=(seq_len, embedding_size)

    pass


if __name__ == '__main__':
    test_elmoformanylangs()