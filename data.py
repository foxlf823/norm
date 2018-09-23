import codecs
from alphabet import Alphabet
import numpy as np
import cPickle as pickle

def loadData(file_path):
    file = codecs.open(file_path, 'r', 'UTF-8')
    data = []
    sentence = []
    for line in file:
        columns = line.strip().split()
        if len(columns) == 0:
            data.append(sentence)
            sentence = []
            continue

        token = {}
        token['text'] = columns[0]
        token['doc'] = columns[1]
        token['start'] = int(columns[2])
        token['end'] = int(columns[3])
        token['label'] = columns[5]
        sentence.append(token)

    file.close()
    return data

def read_instance(data, word_alphabet, char_alphabet, label_alphabet):

    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for sentence in data:
        for token in sentence:
            words.append(token['text'])
            word_Ids.append(word_alphabet.get_index(token['text']))
            labels.append(token['label'])
            label_Ids.append(label_alphabet.get_index(token['label']))
            char_list = []
            char_Id = []
            for char in token['text']:
                char_list.append(char)
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        instence_texts.append([words, chars, labels])
        instence_Ids.append([word_Ids, char_Ids, label_Ids])
        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        label_Ids = []

    return instence_texts, instence_Ids

# def _readString(f):
#     s = str()
#     c = f.read(1).decode('iso-8859-1')
#     while c != '\n' and c != ' ':
#         s = s + c
#         c = f.read(1).decode('iso-8859-1')
#
#     return s

def _readString(f, code):
    s = unicode()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        temp = str()
        temp = temp + c
        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break
                # try:
                #     embedd_dict[word.decode('utf-8')] = word_vector
                # except Exception , e:
                #     pass
                embedd_dict[word] = word_vector
    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
        # with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                # feili
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
                # embedd_dict[tokens[0].decode('utf-8')] = embedd

    return embedd_dict, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim, norm):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


class Data:
    def __init__(self, opt):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)

        self.train_texts = None
        self.train_Ids = None
        self.dev_texts = None
        self.dev_Ids = None
        self.test_texts = None
        self.test_Ids = None

        self.pretrain_word_embedding = None
        self.word_emb_dim = opt.word_emb_dim

    def build_alphabet(self, data):
        for sentence in data:
            for token in sentence:
                self.word_alphabet.add(token['text'])
                self.label_alphabet.add(token['label'])
                for char in token['text']:
                    self.char_alphabet.add(char)

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()








