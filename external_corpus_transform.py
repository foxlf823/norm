import os
from options import opt
import nltk
import logging
from data import processOneFile

# applicable to made, cardio
def load_corpus_bioc_txt_style(basedir, types, type_filter):
    list_dir = os.listdir(basedir)
    if 'bioc' in list_dir:
        annotation_dir = os.path.join(basedir, 'bioc')
    elif 'annotations' in list_dir:
        annotation_dir = os.path.join(basedir, 'annotations')
    else:
        raise RuntimeError("no bioc or annotations in {}".format(basedir))

    if 'txt' in list_dir:
        corpus_dir = os.path.join(basedir, 'txt')
    elif 'corpus' in list_dir:
        corpus_dir = os.path.join(basedir, 'corpus')
    else:
        raise RuntimeError("no txt or corpus in {}".format(basedir))

    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise RuntimeError("invalid nlp tool")

    documents = []

    count_document = 0
    count_sentence = 0
    count_entity = 0

    annotation_files = [f for f in os.listdir(annotation_dir) if os.isfile(os.path.join(annotation_dir, f))]
    for fileName in annotation_files:
        try:
            document = processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool, types, type_filter)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.append(document)

        # statistics
        count_document += 1
        count_sentence += len(document.sentences)
        count_entity += len(document.entities)


    logging.info("document number: {}".format(count_document))
    logging.info("sentence number: {}".format(count_sentence))
    logging.info("entity number {}".format(count_entity))

    return documents
