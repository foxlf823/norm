from seqmodel import SeqModel
import torch
import os
from my_utils import evaluate, makedir_and_clear
import codecs
from os.path import isfile, join
from os import listdir
import spacy
from data import processOneFile, read_instance_from_one_document
import bioc
from data_structure import Entity
import logging
import nltk
from my_corenlp_wrapper import StanfordCoreNLP

def checkWrongState(labelSequence):
    positionNew = -1
    positionOther = -1
    currentLabel = labelSequence[-1]
    assert currentLabel[0] == 'M' or currentLabel[0] == 'E'

    for j in range(len(labelSequence)-1)[::-1]:
        if positionNew == -1 and currentLabel[2:] == labelSequence[j][2:] and labelSequence[j][0] == 'B' :
            positionNew = j
        elif positionOther == -1 and (currentLabel[2:] != labelSequence[j][2:] or labelSequence[j][0] != 'M'):
            positionOther = j

        if positionOther != -1 and positionNew != -1:
            break

    if positionNew == -1:
        return False
    elif positionOther < positionNew:
        return True
    else:
        return False

def translateResultsintoEntities(sentences, predict_results, doc_name):

    entity_id = 1
    results = []

    sent_num = len(predict_results)
    for idx in range(sent_num):
        sent_length = len(predict_results[idx])
        sent_token = sentences[idx]

        assert len(sent_token) == sent_length, "file {}, sent {}".format(doc_name, idx)
        labelSequence = []

        for idy in range(sent_length):
            token = sent_token[idy]
            label = predict_results[idx][idy]
            labelSequence.append(label)

            if label[0] == 'S' or label[0] == 'B':
                entity = Entity()
                entity.create(str(entity_id), label[2:], token['start'], token['end'], token['text'], idx, idy, idy)
                results.append(entity)
                entity_id += 1

            elif label[0] == 'M' or label[0] == 'E':
                if checkWrongState(labelSequence):
                    entity = results[-1]
                    entity.append(token['start'], token['end'], token['text'], idy)


    return results


def dump_results(doc_name, entities, opt):
    collection = bioc.BioCCollection()
    document = bioc.BioCDocument()
    collection.add_document(document)
    document.id = doc_name
    passage = bioc.BioCPassage()
    document.add_passage(passage)
    passage.offset = 0

    for entity in entities:
        anno_entity = bioc.BioCAnnotation()
        passage.add_annotation(anno_entity)
        anno_entity.id = entity.id
        anno_entity.infons['type'] = entity.type
        anno_entity_location = bioc.BioCLocation(entity.start, entity.getlength())
        anno_entity.add_location(anno_entity_location)
        anno_entity.text = entity.text

    with codecs.open(os.path.join(opt.predict, doc_name + ".bioc.xml"), 'w', 'UTF-8') as fp:
        bioc.dump(collection, fp)



def test(data, opt):
    # corpus_dir = join(opt.test_file, 'corpus')
    corpus_dir = join(opt.test_file, 'txt')

    if opt.nlp_tool == "spacy":
        nlp_tool = spacy.load('en')
    elif opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    elif opt.nlp_tool == "stanford":
        nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise RuntimeError("invalid nlp tool")

    corpus_files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]

    model = SeqModel(data, opt)
    model.load_state_dict(torch.load(os.path.join(opt.output, 'model.pkl')))

    makedir_and_clear(opt.predict)

    for fileName in corpus_files:
        try:
            document = processOneFile(fileName, None, corpus_dir, nlp_tool)

            data.test_texts = []
            data.test_Ids = []
            read_instance_from_one_document(document, data.word_alphabet, data.char_alphabet, data.label_alphabet,
                                            data.test_texts, data.test_Ids)

            _, _, _, _, _, pred_results, _ = evaluate(data, opt, model, 'test', False, opt.nbest)

            entities = translateResultsintoEntities(document.sentences, pred_results, fileName)

            dump_results(fileName, entities, opt)
        except Exception, e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue



    logging.info("test finished")