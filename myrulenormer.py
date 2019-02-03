
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

def add_token(s, token):
    if token not in stop_word:
        token = token.lower()

        # replace token with its lemma
        token = wnl.lemmatize(token)

        s.add(token)

def preprocess(str, useBackground):

    tokens = FoxTokenizer.tokenize(0, str, True)
    tokens1 = set()
    for token in tokens:
        if len(token) < 2:
            continue

        # for abbr token, replace it with its full name
        if token in abbr:
            full_names = abbr[token]
            for full_name in full_names:
                t1s = FoxTokenizer.tokenize(0, full_name, True)
                for t1 in t1s:
                    add_token(tokens1, t1)

            continue

        add_token(tokens1, token)


    # add token's background word
    if useBackground:
        tokens2 = set()
        for token1 in tokens1:
            if token1 in background:
                names = background[token1]
                for name in names:
                    t2s = FoxTokenizer.tokenize(0, name, True)
                    for t2 in t2s:
                        add_token(tokens2, t2)
        tokens1 = tokens1 | tokens2

    # add tokens' context word


    return tokens1

def dict_refine(str):
    return re.sub(r'\bNOS\b|\bfinding\b|\(.+?\)|\[.+?\]|\bunspecified\b', ' ', str, flags=re.I).strip()

def make_dictionary(d) :
    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])
    logging.info("dict concept number {}".format(len(UMLS_dict)))

    fp = codecs.open("dictionary.txt", 'w', 'UTF-8')

    for cui, concept in UMLS_dict.items():
        new_names = set()
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
            new_name = preprocess(name, True)
            # all synonym merged
            new_names = new_names | new_name

        for i, name in enumerate(new_names):
            if i == len(new_names)-1:
                write_str += name
            else:
                write_str += name + ','



        fp.write(write_str+"\n")


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

        UMLS_dict[columns[0]] = concept


    fp.close()

    return UMLS_dict


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

def process_one_doc(document, entities, dictionary):

    for entity in entities:

        # if entity.name == 'Dental difficulties':
        #     logging.info(1)
        #     pass

        entity_tokens = preprocess(entity.name, True)



        max_cui = compare(entity_tokens, dictionary)

        for cui in max_cui:
            concept = dictionary[cui]
            entity.norm_ids.append(cui)
            # entity.norm_names.append(concept.names)


def determine_norm_result(gold_entity, predict_entity):

    for norm_id in predict_entity.norm_ids:
        if norm_id in dictionary:
            concept = dictionary[norm_id]

            if gold_entity.norm_ids[0] in concept.codes:

                return True

    return False


if __name__ == '__main__':

    logging.info(opt)
    d = Data(opt)
    logging.info(d.config)

    if opt.whattodo == 1:
        make_dictionary(d)

    else:
        dictionary = load_dictionary(d)

        documents = loadData(opt.test_file, False, opt.types, opt.type_filter)

        ct_predicted = 0
        ct_gold = 0
        ct_correct = 0

        for document in documents:
            logging.debug("###### begin {}".format(document.name))

            pred_entities = []
            for gold in document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.name = gold.name
                pred_entities.append(pred)

            process_one_doc(document, pred_entities, dictionary)

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

            logging.debug("###### end {}, gold {}, predict {}, correct {}".format(document.name, ct_norm_gold, ct_norm_predict, ct_norm_correct))

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







