
from options import opt
from my_utils import normalize_word
from data import my_tokenize
from data_structure import Entity
import multi_sieve

def word_preprocess(word):
    if opt.number_normalized:
        word = normalize_word(word)
    word = word.lower()
    return word


def build_alphabet(alphabet, data):
    for document in data:
        for sentence in document.sentences:
            for token in sentence:
                word = token['text']
                alphabet.add(word_preprocess(word))

def open_alphabet(alphabet):
    alphabet.open()

def fix_alphabet(alphabet):
    alphabet.close()

def build_alphabet_from_dict(alphabet, meddra_dict):
    for concept_id, concept_name in meddra_dict.items():
        tokens = my_tokenize(concept_name)
        for word in tokens:
            alphabet.add(word_preprocess(word))

def get_dict_index(dict_alphabet, concept_id):
    index = dict_alphabet.get_index(concept_id)-2 # since alphabet begin at 2
    return index

def get_dict_name(dict_alphabet, concept_index):
    name = dict_alphabet.get_instance(concept_index+2)
    return name

def init_dict_alphabet(dict_alphabet, meddra_dict):

    for concept_id, concept_name in meddra_dict.items():
        dict_alphabet.add(concept_id)

def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-2


def evaluate(documents, meddra_dict, model):
    if model is not None:
        model.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for document in documents:

        # copy entities from gold entities
        pred_entities = []
        for gold in document.entities:
            pred = Entity()
            pred.id = gold.id
            pred.type = gold.type
            pred.spans = gold.spans
            pred.section = gold.section
            pred.name = gold.name
            pred_entities.append(pred)

        if model is not None:
            model.process_one_doc(document, pred_entities, meddra_dict)
        else:
            multi_sieve.runMultiPassSieve(document, pred_entities, meddra_dict)

        ct_gold += len(document.entities)
        ct_predicted += len(pred_entities)
        for idx, pred in enumerate(pred_entities):
            gold = document.entities[idx]
            if len(pred.norm_ids) != 0 and pred.norm_ids[0] in gold.norm_ids:
                ct_correct += 1


    if ct_gold == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure