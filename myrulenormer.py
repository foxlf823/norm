
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

    return tokens3

def dict_refine(str):
    return re.sub(r'\bNOS\b|\bfinding\b|\(.+?\)|\[.+?\]|\bunspecified\b', ' ', str, flags=re.I).strip()

def make_dictionary(d) :
    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])
    logging.info("dict concept number {}".format(len(UMLS_dict)))

    fp = codecs.open("dictionary.txt", 'w', 'UTF-8')

    for cui, concept in UMLS_dict.items():
        new_names = set()
        # new_names = SortedSet()
        write_str = cui+'|'
        for i, id in enumerate(concept.codes):
            if i == len(concept.codes)-1:
                write_str += id+'|'
            else:
                write_str += id + ','


        for name in concept.names:
            # replace (finding), NOS to whitespace
            name = dict_refine(name)

            # given a name, output its token set
            new_name = preprocess(name, True, False)
            # all synonym merged
            new_names = new_names | new_name

        for i, name in enumerate(new_names):
            if i == len(new_names)-1:
                write_str += name
            else:
                write_str += name + ','



        fp.write(write_str+"\n")


    fp.close()

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

def load_dictionary(d):
    UMLS_dict = {}
    fp = codecs.open("dictionary.txt", 'r', 'UTF-8')

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
            entity_tokens = preprocess(entity.name, True, True)
            entity_key = ""
            for token in entity_tokens:
                entity_key += token + "_"

            if entity_key in train_annotations_dict:
                cui_list = train_annotations_dict[entity_key]
                for cui in cui_list:
                    entity.norm_ids.append(cui)

                continue

        entity_tokens = preprocess(entity.name, True, False)
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

            entity_tokens = preprocess(gold_entity.name, True, True)
            entity_key = ""
            for token in entity_tokens:
                entity_key += token+"_"

            if gold_entity.norm_ids[0] in dictionary_reverse:
                cui_list = dictionary_reverse[gold_entity.norm_ids[0]]

                for cui in cui_list:
                    setMap(train_annotations_dict, entity_key, cui)

    return train_annotations_dict




if __name__ == '__main__':

    logging.info(opt)
    d = Data(opt)
    logging.info(d.config)

    if opt.whattodo == 1:
        make_dictionary(d)

    else:


        documents = loadData(opt.test_file, False, opt.types, opt.type_filter)

        if opt.train_file:
            logging.info("use train data")
            dictionary_reverse = load_dictionary_reverse()
            train_annotations_dict = load_train_set(dictionary_reverse)
        else:
            train_annotations_dict = None

        dictionary = load_dictionary(d)

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







