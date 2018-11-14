import codecs
from alphabet import Alphabet
import numpy as np
import pickle as pk
from os import listdir
from os.path import isfile, join
from my_utils import get_bioc_file, get_text_file, normalize_word
import spacy
from data_structure import Entity, Document
from options import opt
import logging
import re
import nltk
from my_corenlp_wrapper import StanfordCoreNLP
import json

def getLabel(start, end, entities):
    match = ""
    for entity in entities:
        if start == entity.start and end == entity.end : # S
            match = "S"
            break
        elif start == entity.start and end != entity.end : # B
            match = "B"
            break
        elif start != entity.start and end == entity.end : # E
            match = "E"
            break
        elif start > entity.start and end < entity.end:  # M
            match = "M"
            break

    if match != "":
        if opt.no_type:
            return match + "-" +"X"
        else:
            return match+"-"+entity.type
    else:
        return "O"

def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp, entities):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
            # if entities:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

# if add pos, add to the end, so external functions don't need to be modified too much
# def text_tokenize_and_postagging(txt, sent_start):
#     tokens= my_tokenize(txt)
#     pos_tags = nltk.pos_tag(tokens)
#
#     offset = 0
#     for token, pos_tag in pos_tags:
#         offset = txt.find(token, offset)
#         yield token, pos_tag, offset+sent_start, offset+len(token)+sent_start
#         offset += len(token)

def text_tokenize_and_postagging(txt, sent_start):
    tokens= my_tokenize(txt)

    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start
        offset += len(token)

def token_from_sent(txt, sent_start):
    return [token for token in text_tokenize_and_postagging(txt, sent_start)]

def get_sentences_and_tokens_from_nltk(text, nlp_tool, entities):
    all_sents_inds = []
    generator = nlp_tool.span_tokenize(text)
    for t in generator:
        all_sents_inds.append(t)

    sentences = []
    for ind in range(len(all_sents_inds)):
        t_start = all_sents_inds[ind][0]
        t_end = all_sents_inds[ind][1]
        tmp_tokens = token_from_sent(text[t_start:t_end], t_start)
        sentence_tokens = []
        for token in tmp_tokens:
            token_dict = {}
            token_dict['start'], token_dict['end'] = token[1], token[2]
            token_dict['text'] = token[0]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    text = text.encode("utf-8")
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    # if type(output) is str:
    if type(output) is unicode:
        output = json.loads(output, strict=False)
    return output

def get_sentences_and_tokens_from_stanford(text, nlp_tool, entities):
    stanford_output = get_stanford_annotations(text, nlp_tool)
    sentences = []
    temp = stanford_output['sentences']
    for sentence in stanford_output['sentences']:
        sentence_tokens = []
        for stanford_token in sentence['tokens']:
            token_dict = {}
            token_dict['start'] = int(stanford_token['characterOffsetBegin'])
            token_dict['end'] = int(stanford_token['characterOffsetEnd'])
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                                     token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences


def processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool):
    document = Document()
    document.name = fileName[:fileName.find('.')]

    if annotation_dir:
        annotation_file = get_bioc_file(join(annotation_dir, fileName))
        bioc_passage = annotation_file[0].passages[0]
        entities = []

        for entity in bioc_passage.annotations:
            if opt.types and (entity.infons['type'] not in opt.type_filter):
                continue
            entity_ = Entity()
            if isinstance(entity.text, str):
                # text = entity.text.decode('utf-8')
                text = entity.text
            else: # unicode
                text = entity.text
            entity_.create(entity.id, entity.infons['type'], entity.locations[0].offset, entity.locations[0].end,
                           text, None, None, None)
            # try:
            #     entity_.create(entity.id, entity.infons['type'], entity.locations[0].offset, entity.locations[0].end,
            #                entity.text.decode('utf-8'), None, None, None)
            # except Exception, e:
            #     print("entity.text.decode error: document:{}, entity.id:{}".format(fileName, entity.id))
            #     continue
            entities.append(entity_)

        document.entities = entities

    corpus_file = get_text_file(join(corpus_dir, fileName.split('.bioc')[0]))

    if opt.nlp_tool == "spacy":
        if annotation_dir:
            sentences = get_sentences_and_tokens_from_spacy(corpus_file, nlp_tool, document.entities)
        else:
            sentences = get_sentences_and_tokens_from_spacy(corpus_file, nlp_tool, None)
    elif opt.nlp_tool == "nltk":
        if annotation_dir:
            sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, document.entities)
        else:
            sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, None)
    elif opt.nlp_tool == "stanford":
        if annotation_dir:
            sentences = get_sentences_and_tokens_from_stanford(corpus_file, nlp_tool, document.entities)
        else:
            sentences = get_sentences_and_tokens_from_stanford(corpus_file, nlp_tool, None)
    else:
        raise RuntimeError("invalid nlp tool")


    document.sentences = sentences

    return document



def loadData(basedir):
    # annotation_dir = join(basedir, 'annotations')
    # corpus_dir = join(basedir, 'corpus')

    annotation_dir = join(basedir, 'bioc')
    corpus_dir = join(basedir, 'txt')
    # spacy, nltk, stanford
    if opt.nlp_tool == "spacy":
        nlp_tool = spacy.load('en')
    elif opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    elif opt.nlp_tool == "stanford":
        nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise RuntimeError("invalid nlp tool")

    documents = []

    count_entity_mention = 0

    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]
    for fileName in annotation_files:
        try:
            document = processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.append(document)

        count_entity_mention += len(document.entities)

    logging.info("{} entities in {}".format(count_entity_mention, basedir))

    return documents

def read_instance_from_one_document(document, word_alphabet, char_alphabet, label_alphabet, instence_texts, instence_Ids):

    for sentence in document.sentences:

        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        label_Ids = []

        for token in sentence:
            word = token['text']
            if opt.number_normalized:
                word = normalize_word(word)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
            if 'label' in token:
                labels.append(token['label'])
                label_Ids.append(label_alphabet.get_index(token['label']))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        if len(labels) == 0:
            instence_texts.append([words, chars])
            instence_Ids.append([word_Ids, char_Ids])
        else:
            instence_texts.append([words, chars, labels])
            instence_Ids.append([word_Ids, char_Ids, label_Ids])


def read_instance(data, word_alphabet, char_alphabet, label_alphabet):

    instence_texts = []
    instence_Ids = []

    for document in data:
        read_instance_from_one_document(document, word_alphabet, char_alphabet, label_alphabet, instence_texts,
                                        instence_Ids)

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
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
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
        # elif re.sub('\d', '0', word) in embedd_dict:
        #     if norm:
        #         pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
        #     else:
        #         pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
        #     digits_replaced_with_zeros_found += 1
        # elif re.sub('\d', '0', word.lower()) in embedd_dict:
        #     if norm:
        #         pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
        #     else:
        #         pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
        #     lowercase_and_digits_replaced_with_zeros_found += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logging.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, dig_zero_match:%s, "
                 "case_dig_zero_match:%s, oov:%s, oov%%:%s"
                 %(pretrained_size, perfect_match, case_match, digits_replaced_with_zeros_found,
                   lowercase_and_digits_replaced_with_zeros_found, not_match, (not_match+0.)/alphabet_size))
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

    def clear(self):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.train_texts = None
        self.train_Ids = None
        self.dev_texts = None
        self.dev_Ids = None
        self.test_texts = None
        self.test_Ids = None

        self.pretrain_word_embedding = None


    def build_alphabet(self, data):
        for document in data:
            for sentence in document.sentences:
                for token in sentence:
                    word = token['text']
                    if opt.number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)
                    self.label_alphabet.add(token['label'])
                    # try:
                    #     self.label_alphabet.add(token['label'])
                    # except Exception, e:
                    #     print("document id {} {} {}".format(document.name))
                    #     exit()
                    for char in word:
                        self.char_alphabet.add(char)


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pk.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pk.dump(self.__dict__, f, 2)
        f.close()








