
from data import Data, loadData
from options import opt
import logging
logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
from fox_tokenizer import FoxTokenizer
from stopword import stop_word
import umls
from my_utils import setList, setMap
import codecs
import re
from data_structure import Entity
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
MIN_WORD_LEN = 2
from sortedcontainers import SortedSet

from alphabet import Alphabet
from data import build_pretrain_embedding
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as functional
from my_utils import random_embedding, freeze_net

import numpy as np
import time
import os
from os import listdir
from os.path import isfile, join

from evaluate_metamap import parse_one_gold_file
import metamap
import multi_sieve
import copy


def getAbbr_fromFile(file_path):
    s = dict()

    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp:

            line = line.strip()
            if line == u'':
                continue

            columns = line.split('||')
            # abbr is cased, full name is uncased
            # one abbr may have multiple full names
            setMap(s, columns[0], columns[1].lower())

    return s

abbr = getAbbr_fromFile('abbreviations.txt')

def getBackground_fromFile(file_path):
    s = dict()

    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp:

            line = line.strip()
            if line == u'':
                continue

            columns = line.split('|')

            # for each word in the column, we create a dict item for retrieval, uncased
            for i, a in enumerate(columns):
                for j, b in enumerate(columns):
                    if i==j:
                        continue

                    setMap(s, columns[i].lower(), columns[j].lower())

    return s

background = getBackground_fromFile('background.txt')



def preprocess(str, useBackground, useSortedSet):

    tokens = FoxTokenizer.tokenize(0, str, True)

    tokens1 = set()
    for token in tokens:
        # word len
        if len(token) < MIN_WORD_LEN:
            continue

        # replace abbr, cased, one abbr -> many full names
        if token in abbr:
            full_names = abbr[token]
            for full_name in full_names:
                t1s = FoxTokenizer.tokenize(0, full_name, True)
                for t1 in t1s:
                    tokens1.add(t1.lower())
        else:
            tokens1.add(token)

    tokens2 = set()
    for token in tokens1:
        # lower
        token = token.lower()

        # stop word
        if token in stop_word:
            continue

        # lemma
        token = wnl.lemmatize(token)

        tokens2.add(token)

        # add background
        if useBackground and token in background:
            names = background[token]
            for name in names:
                t2s = FoxTokenizer.tokenize(0, name, True)
                for t2 in t2s:
                    tokens2.add(t2.lower())

    if useSortedSet:
        tokens3 = SortedSet()
    else:
        tokens3 = set()
    for token in tokens2:
        token = lancaster.stem(token)

        # word len
        if len(token) < MIN_WORD_LEN:
            continue

        tokens3.add(token)

    return tokens3, tokens2

def dict_refine(str):
    return re.sub(r'\bNOS\b|\bfinding\b|\(.+?\)|\[.+?\]|\bunspecified\b', ' ', str, flags=re.I).strip()

def make_dictionary(d) :
    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])
    logging.info("dict concept number {}".format(len(UMLS_dict)))

    fp = codecs.open("dictionary.txt", 'w', 'UTF-8')
    fp1 = codecs.open("dictionary_full.txt", 'w', 'UTF-8')

    for cui, concept in UMLS_dict.items():
        new_names = set()
        new_names_full = set()
        write_str = cui+'|'
        write_str1 = cui+'|'
        for i, id in enumerate(concept.codes):
            if i == len(concept.codes)-1:
                write_str += id+'|'
                write_str1 += id+'|'
            else:
                write_str += id + ','
                write_str1 += id + ','


        for name in concept.names:
            # replace (finding), NOS to whitespace
            name = dict_refine(name)

            # given a name, output its token set
            new_name, new_name_full = preprocess(name, True, False)
            if len(new_name) == 0 or len(new_name_full) == 0:
                raise RuntimeError("empty after preprocess: {}".format(name))
            # all synonym merged
            new_names = new_names | new_name
            new_names_full = new_names_full | new_name_full

        for i, name in enumerate(new_names):
            if i == len(new_names)-1:
                write_str += name
            else:
                write_str += name + ','

        for i, name in enumerate(new_names_full):
            if i == len(new_names_full) - 1:
                write_str1 += name
            else:
                write_str1 += name + ','



        fp.write(write_str+"\n")
        fp1.write(write_str1+"\n")


    fp.close()
    fp1.close()

    fp = codecs.open("dictionary_reverse.txt", 'w', 'UTF-8')

    for code, cui_list in UMLS_dict_reverse.items():
        write_str = code + '|'
        for i, cui in enumerate(cui_list):
            if i == len(cui_list)-1:
                write_str += cui
            else:
                write_str += cui + ','

        fp.write(write_str + "\n")

    fp.close()

def load_dictionary(path):
    UMLS_dict = {}
    fp = codecs.open(path, 'r', 'UTF-8')

    for line in fp:
        line = line.strip()
        columns = line.split('|')
        concept = umls.UMLS_Concept()
        concept.cui = columns[0]
        concept.codes = list(columns[1].split(','))
        concept.names = set(columns[2].split(','))
        # concept.names = SortedSet(columns[2].split(','))

        UMLS_dict[columns[0]] = concept


    fp.close()

    return UMLS_dict

def load_dictionary_reverse():
    dictionary_reverse = {}

    fp = codecs.open("dictionary_reverse.txt", 'r', 'UTF-8')

    for line in fp:
        line = line.strip()
        columns = line.split('|')
        dictionary_reverse[columns[0]] = list(columns[1].split(','))

    fp.close()

    return dictionary_reverse



def compare(gold_entity_tokens, dictionary):

    max_cui = [] # there may be more than one
    max_number = 1

    for cui, concept in dictionary.items():
        intersect_tokens = gold_entity_tokens & concept.names
        intersect_number = len(intersect_tokens)

        if intersect_number > max_number:
            max_number = intersect_number
            max_cui.clear()
            max_cui.append(cui)
        elif intersect_number == max_number:
            max_cui.append(cui)

    return max_cui

def process_one_doc(document, entities, dictionary, train_annotations_dict):

    for entity in entities:

        # if entity.name == 'Dental difficulties':
        #     logging.info(1)
        #     pass



        if train_annotations_dict is not None:
            entity_tokens, _ = preprocess(entity.name, True, True)
            if len(entity_tokens) != 0:

                entity_key = ""
                for token in entity_tokens:
                    entity_key += token + "_"

                if entity_key in train_annotations_dict:
                    cui_list = train_annotations_dict[entity_key]
                    for cui in cui_list:
                        entity.norm_ids.append(cui)

                    continue

        entity_tokens, _ = preprocess(entity.name, True, False)
        if len(entity_tokens) == 0:
            continue

        max_cui = compare(entity_tokens, dictionary)

        for cui in max_cui:
            # concept = dictionary[cui]
            entity.norm_ids.append(cui)
            # entity.norm_names.append(concept.names)


def determine_norm_result(gold_entity, predict_entity):

    for norm_id in predict_entity.norm_ids:
        if norm_id in dictionary:
            concept = dictionary[norm_id]

            if gold_entity.norm_ids[0] in concept.codes:

                return True

    return False

# if we have train set, use the annotations directly
# its priority is higher than rules
def load_train_set(dictionary_reverse):

    documents = loadData(opt.train_file, False, opt.types, opt.type_filter)

    train_annotations_dict = {}

    for document in documents:
        for gold_entity in document.entities:

            entity_tokens, _ = preprocess(gold_entity.name, True, True)
            if len(entity_tokens) == 0:
                continue

            entity_key = ""
            for token in entity_tokens:
                entity_key += token+"_"

            if gold_entity.norm_ids[0] in dictionary_reverse:
                cui_list = dictionary_reverse[gold_entity.norm_ids[0]]

                for cui in cui_list:
                    setMap(train_annotations_dict, entity_key, cui)

    return train_annotations_dict


def load_dataponts(path):
    fp = codecs.open(path, 'r', 'UTF-8')
    datapoints = []
    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue

        columns = line.split('|')

        one_datapoint = [] # 0-mention, 1-gold, other-neg
        for i, column in enumerate(columns):
            words = column.split(',')
            one_datapoint.append(words)

        datapoints.append(one_datapoint)

    return datapoints


def build_alphabet_from_dict(alphabet, dictionary):

    for concept_id, concept in dictionary.items():
        for word in concept.names:
            alphabet.add(word)

def build_alphabet(alphabet, datapoints):
    for datapoint in datapoints:
        for column in datapoint:
            for word in column:
                alphabet.add(word)

class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if opt.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output

class VsmNormer(nn.Module):

    def __init__(self, word_alphabet, word_embedding, embedding_dim):
        super(VsmNormer, self).__init__()
        self.word_alphabet = word_alphabet
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.gpu = opt.gpu
        self.margin = 1
        self.word_drop = nn.Dropout(opt.dropout)
        self.attn = DotAttentionLayer(self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear.weight.data.copy_(torch.eye(self.embedding_dim))

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.attn = self.attn.cuda(self.gpu)
            self.linear = self.linear.cuda(self.gpu)


    # mention (1,mention_length), concepts (n, concepts_length)
    def forward(self, mention, mention_length, concepts, concepts_length):

        mention_word_emb = self.word_embedding(mention)
        mention_word_emb = self.word_drop(mention_word_emb)
        mention_rep = self.attn((mention_word_emb, mention_length))

        concept_word_emb = self.word_embedding(concepts)
        concept_word_emb = self.word_drop(concept_word_emb)
        concept_rep = self.attn((concept_word_emb, concepts_length))

        m_W = self.linear(mention_rep)

        similarity = torch.matmul(m_W, torch.t(concept_rep))

        return similarity

    def loss_function(self, similarity, y):
        concept_size = similarity.size(-1)

        y_expand = y.unsqueeze(-1).expand(-1, concept_size)

        a = torch.gather(similarity, 1, y_expand)

        loss = (self.margin - a + similarity).clamp(min=0)

        batch_size = similarity.size(0)
        one_hot = torch.zeros(batch_size, concept_size)
        if opt.gpu >= 0 and torch.cuda.is_available():
            one_hot = one_hot.cuda(opt.gpu)

        one_hot = one_hot.scatter_(1, y.unsqueeze(-1), self.margin)

        loss = torch.sum(loss - one_hot) / batch_size

        return loss

def generate_instances(word_alphabet, datapoints):
    instances = []
    for datapoint in datapoints:
        one_instance = []
        for entity in datapoint:
            id_list = []
            for token in entity:
                word_id = word_alphabet.get_index(token)
                id_list.append(word_id)

            one_instance.append(id_list)

        instances.append(one_instance)

    return instances

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def my_collate(x): # batch_size = 1
    x = x[0]
    # x[0]-mention x[1]-gold other-neg
    mention = x[0]
    mention_length = len(mention)
    mention = pad_sequence([mention], mention_length)
    mention_length = torch.LongTensor([mention_length])

    concepts = x[1:]
    concepts_length = [len(row) for row in concepts]
    max_len = max(concepts_length)
    concepts = pad_sequence(concepts, max_len)
    concepts_length = torch.LongTensor(concepts_length)

    y = torch.LongTensor([0]) # gold is the 0-th item in concepts

    if opt.gpu >= 0 and torch.cuda.is_available():
        mention = mention.cuda(opt.gpu)
        mention_length = mention_length.cuda(opt.gpu)
        concepts = concepts.cuda(opt.gpu)
        concepts_length = concepts_length.cuda(opt.gpu)
        y = y.cuda(opt.gpu)

    return mention, mention_length, concepts, concepts_length, y

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x


if __name__ == '__main__':

    logging.info(opt)
    d = Data(opt)
    logging.info(d.config)

    if opt.whattodo == 1: # make dictionary
        make_dictionary(d)

    elif opt.whattodo == 2: # rulenormer, evaluate is loose, can be see as upper boundanry


        documents = loadData(opt.test_file, False, opt.types, opt.type_filter)

        if opt.train_file:
            logging.info("use train data")
            dictionary_reverse = load_dictionary_reverse()
            train_annotations_dict = load_train_set(dictionary_reverse)
        else:
            train_annotations_dict = None

        dictionary = load_dictionary("dictionary.txt")

        ct_predicted = 0
        ct_gold = 0
        ct_correct = 0

        # stat
        ct_answer_zero = 0
        ct_answer_one = 0
        ct_answer_multi_gold_in = 0
        ct_answer_multi_gold_not_in = 0

        for document in documents:
            logging.info("###### begin {}".format(document.name))

            pred_entities = []
            for gold in document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.name = gold.name
                pred_entities.append(pred)

            process_one_doc(document, pred_entities, dictionary, train_annotations_dict)

            ct_norm_gold = len(document.entities)
            ct_norm_predict = len(pred_entities)
            ct_norm_correct = 0


            for predict_entity in pred_entities:

                for gold_entity in document.entities:

                    if predict_entity.equals_span(gold_entity):


                        b_right = False

                        if determine_norm_result(gold_entity, predict_entity):
                            ct_norm_correct += 1
                            b_right = True

                        # stat
                        if len(predict_entity.norm_ids) == 0:
                            ct_answer_zero += 1
                        elif len(predict_entity.norm_ids) == 1:
                            ct_answer_one += 1
                        else:
                            find1 = False
                            for norm_id in predict_entity.norm_ids:
                                if norm_id in dictionary:
                                    concept = dictionary[norm_id]

                                    if gold_entity.norm_ids[0] in concept.codes:
                                        find1 = True
                                        break
                            if find1:
                                ct_answer_multi_gold_in += 1
                            else:
                                ct_answer_multi_gold_not_in += 1




                        if b_right == False:

                            if len(predict_entity.norm_ids) == 0:
                                logging.debug("### entity norm failed: {}".format(predict_entity.name))
                                logging.debug(
                                    "entity name: {} | gold id, name: {}, {} | pred cui, codes, names: , , "
                                    .format(predict_entity.name, gold_entity.norm_ids[0],
                                            gold_entity.norm_names[0]))
                            else:
                                logging.debug("### entity norm wrong: {}".format(predict_entity.name))
                                for norm_id in predict_entity.norm_ids:
                                    if norm_id in dictionary:
                                        concept = dictionary[norm_id]

                                        logging.debug("entity name: {} | gold id, name: {}, {} | pred cui, codes, names: {}, {}, {}"
                                                .format(predict_entity.name, gold_entity.norm_ids[0],
                                                        gold_entity.norm_names[0],
                                                        concept.cui, concept.codes, concept.names))



                        break

            ct_predicted += ct_norm_predict
            ct_gold += ct_norm_gold
            ct_correct += ct_norm_correct

            logging.info("###### end {}, gold {}, predict {}, correct {}".format(document.name, ct_norm_gold, ct_norm_predict, ct_norm_correct))

        if ct_gold == 0:
            precision = 0
            recall = 0
        else:
            precision = ct_correct * 1.0 / ct_predicted
            recall = ct_correct * 1.0 / ct_gold

        if precision + recall == 0:
            f_measure = 0
        else:
            f_measure = 2 * precision * recall / (precision + recall)

        logging.info("p: %.4f, r: %.4f, f: %.4f" % (precision, recall, f_measure))

        # stat
        logging.info("ct_answer_zero {}, ct_answer_one {}, ct_answer_multi_gold_in {}, ct_answer_multi_gold_not_in {}"
                     .format(ct_answer_zero, ct_answer_one, ct_answer_multi_gold_in, ct_answer_multi_gold_not_in))


    elif opt.whattodo == 3: # make train and test instance

        documents = loadData(opt.train_file, False, opt.types, opt.type_filter)

        dictionary = load_dictionary("dictionary.txt")
        dictionary_full = load_dictionary("dictionary_full.txt")
        dictionary_reverse = load_dictionary_reverse()

        training_instances_fp = codecs.open("training_instances.txt", 'w', 'UTF-8')

        for document in documents:
            logging.info("###### begin {}".format(document.name))

            pred_entities = []
            for gold in document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.name = gold.name
                pred_entities.append(pred)

            process_one_doc(document, pred_entities, dictionary, None)

            for predict_entity in pred_entities:

                for gold_entity in document.entities:

                    if predict_entity.equals_span(gold_entity):

                        # b_right = False
                        #
                        # if determine_norm_result(gold_entity, predict_entity):
                        #     b_right = True

                        # stat
                        if len(predict_entity.norm_ids) > 1:

                            write_line = ""

                            _, entity_tokens_full = preprocess(predict_entity.name, True, False)
                            if len(entity_tokens_full) == 0:
                                break

                            for i, token in enumerate(entity_tokens_full):
                                if i == len(entity_tokens_full) - 1:
                                    write_line += token
                                else:
                                    write_line += token + ','

                            write_line += '|'

                            find1 = False
                            find1_idx = 9999
                            for i, norm_id in enumerate(predict_entity.norm_ids):
                                if norm_id in dictionary:
                                    concept = dictionary[norm_id]

                                    if gold_entity.norm_ids[0] in concept.codes:
                                        find1 = True
                                        find1_idx = i
                                        break
                            if find1:

                                concept = dictionary_full[predict_entity.norm_ids[find1_idx]]
                                for i, name in enumerate(concept.names):
                                    if i == len(concept.names) - 1:
                                        write_line += name
                                    else:
                                        write_line += name + ','

                                write_line += '|'

                                for i, norm_id in enumerate(predict_entity.norm_ids):
                                    if i == find1_idx:
                                        continue

                                    concept = dictionary_full[norm_id]
                                    for j, name in enumerate(concept.names):
                                        if j == len(concept.names) - 1:
                                            write_line += name
                                        else:
                                            write_line += name + ','

                                    if i < len(predict_entity.norm_ids) - 1:
                                        write_line += '|'

                            else:

                                if gold_entity.norm_ids[0] in dictionary_reverse:
                                    cui_list = dictionary_reverse[gold_entity.norm_ids[0]]
                                    concept = dictionary_full[cui_list[0]]

                                    for i, name in enumerate(concept.names):
                                        if i == len(concept.names) - 1:
                                            write_line += name
                                        else:
                                            write_line += name + ','

                                    write_line += '|'

                                    for i, norm_id in enumerate(predict_entity.norm_ids):

                                        concept = dictionary_full[norm_id]
                                        for j, name in enumerate(concept.names):
                                            if j == len(concept.names) - 1:
                                                write_line += name
                                            else:
                                                write_line += name + ','

                                        if i < len(predict_entity.norm_ids) - 1:
                                            write_line += '|'

                            training_instances_fp.write(write_line+"\n")


                        break


        training_instances_fp.close()

    elif opt.whattodo == 4: # train vsm on candidates

        datapoints_train = load_dataponts(opt.train_file)
        datapoints_test = load_dataponts(opt.test_file)
        # datapoints_train = load_dataponts('training_instances_debug.txt')
        # datapoints_test = load_dataponts('test_instances_debug.txt')


        word_alphabet = Alphabet('word')

        build_alphabet(word_alphabet, datapoints_train)
        build_alphabet(word_alphabet, datapoints_test)
        word_alphabet.close()

        if d.config.get('norm_emb') is not None:
            logging.info("load pretrained word embedding ...")
            pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_emb'),
                                                                             word_alphabet,
                                                                             opt.word_emb_dim, False)
            word_embedding = nn.Embedding(word_alphabet.size(), word_emb_dim, padding_idx=0)
            word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
            embedding_dim = word_emb_dim
        else:
            logging.info("randomly initialize word embedding ...")
            word_embedding = nn.Embedding(word_alphabet.size(), d.word_emb_dim, padding_idx=0)
            word_embedding.weight.data.copy_(
                torch.from_numpy(random_embedding(word_alphabet.size(), d.word_emb_dim)))
            embedding_dim = d.word_emb_dim


        vsm_model = VsmNormer(word_alphabet, word_embedding, embedding_dim)

        # generate data points
        instances_train = generate_instances(word_alphabet, datapoints_train)
        instances_test = generate_instances(word_alphabet, datapoints_test)

        # batch size always 1
        train_loader = DataLoader(MyDataset(instances_train), 1, shuffle=True, collate_fn=my_collate)
        test_loader = DataLoader(MyDataset(instances_test), 1, shuffle=False, collate_fn=my_collate)

        optimizer = optim.Adam(vsm_model.parameters(), lr=opt.lr, weight_decay=opt.l2)

        if opt.tune_wordemb == False:
            freeze_net(vsm_model.word_embedding)

        best_acc = -10

        bad_counter = 0

        logging.info("start training ...")

        for idx in range(opt.iter):
            epoch_start = time.time()

            vsm_model.train()

            train_iter = iter(train_loader)
            num_iter = len(train_loader)

            sum_loss = 0

            correct, total = 0, 0

            for i in range(num_iter):

                mention, mention_length, concepts, concepts_length, y = next(train_iter)

                similarity = vsm_model.forward(mention, mention_length, concepts, concepts_length)

                l = vsm_model.loss_function(similarity, y)

                sum_loss += l.item()

                l.backward()

                if opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
                optimizer.step()
                vsm_model.zero_grad()

                total += mention.size(0)
                _, pred = torch.max(similarity, 1)
                correct += (pred == y).sum().item()

            epoch_finish = time.time()
            accuracy = 100.0 * correct / total
            logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
                idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

            # evaluate
            test_iter = iter(test_loader)
            num_iter = len(test_loader)
            correct, total = 0, 0
            vsm_model.eval()

            for i in range(num_iter):
                mention, mention_length, concepts, concepts_length, y = next(test_iter)

                similarity = vsm_model.forward(mention, mention_length, concepts, concepts_length)

                total += mention.size(0)
                _, pred = torch.max(similarity, 1)
                correct += (pred == y).sum().item()

            accuracy = 100.0 * correct / total
            logging.info("epoch: %s evaluate finished. Accuracy %.2f" % (
                idx, accuracy))


            if accuracy > best_acc:
                logging.info("Exceed previous best: %.2f" % (best_acc))

                torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))

                best_acc = accuracy

                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                logging.info('Early Stop!')
                break

        logging.info("train finished")

    elif opt.whattodo == 5: # evaluate on metamap

        # logging.info("load umls ...")
        # UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])

        if opt.train_file:
            logging.info("use train data")
            dictionary_reverse = load_dictionary_reverse()
            train_annotations_dict = load_train_set(dictionary_reverse)
        else:
            train_annotations_dict = None

        predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
        annotation_dir = os.path.join(opt.test_file, 'bioc')
        corpus_dir = os.path.join(opt.test_file, 'txt')
        annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f)) and f.find('.xml') != -1]

        logging.info("load dictionary ... ")
        dictionary = load_dictionary("dictionary.txt")
        dictionary_full = load_dictionary("dictionary_full.txt")


        logging.info("load vsm model ...")
        if opt.test_in_cpu:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
        else:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
        vsm_model.eval()

        ct_norm_predict = 0
        ct_norm_gold = 0
        ct_norm_correct = 0

        for gold_file_name in annotation_files:
            print("# begin {}".format(gold_file_name))
            gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

            predict_document = metamap.load_metamap_result_from_file(
                join(predict_dir, gold_file_name[:gold_file_name.find('.')] + ".field.txt"))

            # copy entities from metamap entities
            pred_entities = []
            # for gold in predict_document.entities:
            for gold in gold_document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.section = gold.section
                pred.name = gold.name
                pred_entities.append(pred)

            process_one_doc(gold_document, pred_entities, dictionary, train_annotations_dict)

            for predict_entity in pred_entities:

                for gold_entity in gold_document.entities:

                    if predict_entity.equals_span(gold_entity):

                        if len(predict_entity.norm_ids) == 0:
                            pass
                        elif len(predict_entity.norm_ids) == 1:
                            # if only one answer is returned by myrulenormer, we directly use it
                            concept = dictionary[predict_entity.norm_ids[0]]

                            if gold_entity.norm_ids[0] in concept.codes:
                                ct_norm_correct += 1

                        else: # if there are multiple answers, we use vsm to disambiguate

                            datapoint = []
                            _, entity_tokens_full = preprocess(predict_entity.name, True, False)

                            if len(entity_tokens_full) == 0:
                                logging.info("empty entity tokens: {}".format(predict_entity.name))
                                break


                            mention = []
                            for i, token in enumerate(entity_tokens_full):
                                mention.append(vsm_model.word_alphabet.get_index(token))

                            datapoint.append(mention)

                            for i, norm_id in enumerate(predict_entity.norm_ids):
                                concept_word = []
                                concept = dictionary_full[norm_id]
                                for j, name in enumerate(concept.names):
                                    concept_word.append(vsm_model.word_alphabet.get_index(name))

                                datapoint.append(concept_word)

                            mention, mention_length, concepts, concepts_length, _ = my_collate([datapoint])

                            similarity = vsm_model.forward(mention, mention_length, concepts, concepts_length)

                            _, pred = torch.max(similarity, 1)

                            concept = dictionary[predict_entity.norm_ids[pred.item()]]

                            if gold_entity.norm_ids[0] in concept.codes:
                                ct_norm_correct += 1

                        break


            ct_norm_gold += len(gold_document.entities)
            ct_norm_predict += len(pred_entities)


        p = ct_norm_correct * 1.0 / ct_norm_predict
        r = ct_norm_correct * 1.0 / ct_norm_gold
        f1 = 2.0 * p * r / (p + r)
        print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))


    elif opt.whattodo == 6:

        logging.info("load umls ...")
        UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])

        if opt.train_file:
            logging.info("use train data")
            dictionary_reverse = load_dictionary_reverse()
            train_annotations_dict = load_train_set(dictionary_reverse)
        else:
            train_annotations_dict = None

        predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
        annotation_dir = os.path.join(opt.test_file, 'bioc')
        corpus_dir = os.path.join(opt.test_file, 'txt')
        annotation_files = [f for f in listdir(annotation_dir) if
                            isfile(join(annotation_dir, f)) and f.find('.xml') != -1]

        logging.info("load dictionary ... ")
        dictionary = load_dictionary("dictionary.txt")
        dictionary_full = load_dictionary("dictionary_full.txt")

        multi_sieve.init(opt, None, d, UMLS_dict, UMLS_dict_reverse, False)

        ct_norm_predict = 0
        ct_norm_gold = 0
        ct_norm_correct = 0

        for gold_file_name in annotation_files:
            print("# begin {}".format(gold_file_name))
            gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

            predict_document = metamap.load_metamap_result_from_file(
                join(predict_dir, gold_file_name[:gold_file_name.find('.')] + ".field.txt"))

            # copy entities from metamap entities
            pred_entities = []
            # for gold in predict_document.entities:
            for gold in gold_document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.section = gold.section
                pred.name = gold.name
                pred_entities.append(pred)

            process_one_doc(gold_document, pred_entities, dictionary, train_annotations_dict)

            for predict_entity in pred_entities:

                for gold_entity in gold_document.entities:

                    if predict_entity.equals_span(gold_entity):

                        if len(predict_entity.norm_ids) == 0:
                            pass
                        elif len(predict_entity.norm_ids) == 1:
                            # if only one answer is returned by myrulenormer, we directly use it
                            concept = dictionary[predict_entity.norm_ids[0]]

                            if gold_entity.norm_ids[0] in concept.codes:
                                ct_norm_correct += 1

                        else:  # if there are multiple answers, we use multi-sieve to disambiguate

                            copy_entity = copy.deepcopy(predict_entity)
                            copy_entity.norm_ids = []

                            multi_sieve.runMultiPassSieve_oneentity(gold_document, copy_entity)

                            concept = dictionary[copy_entity.norm_ids[0]]

                            if gold_entity.norm_ids[0] in concept.codes:
                                ct_norm_correct += 1

                        break

            ct_norm_gold += len(gold_document.entities)
            ct_norm_predict += len(pred_entities)

        multi_sieve.finalize(True)

        p = ct_norm_correct * 1.0 / ct_norm_predict
        r = ct_norm_correct * 1.0 / ct_norm_gold
        f1 = 2.0 * p * r / (p + r)
        print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

    else:
        logging.info("wrong whattodo")




