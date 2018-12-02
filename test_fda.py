import nltk
import os
from seqmodel import SeqModel
import torch
from my_utils import makedir_and_clear, evaluate, is_overlapped
import time
from data import processOneFile_fda, read_instance_from_one_document
import logging
from metric import get_ner_BIOHD_1234
from data_structure import Entity
import xml.dom
import codecs

def span_to_start(entity):
    ret = ""
    for span in entity.spans:
        ret += str(span[0])+","
    return ret[:-1]

def span_to_len(entity):
    ret = ""
    for span in entity.spans:
        ret += str(span[1]-span[0])+","
    return ret[:-1]

def dump_results(doc_name, entities, opt, annotation_file):
    entity_id = 1
    dom1 = xml.dom.getDOMImplementation()
    doc = dom1.createDocument(None, "SubmissionLabel", None)
    xml_SubmissionLabel = doc.documentElement

    xml_Text = doc.createElement('Text')
    xml_SubmissionLabel.appendChild(xml_Text)

    for section in annotation_file.sections:
        xml_Section = doc.createElement('Section')
        xml_Section.setAttribute('id', section.id)
        xml_Section.setAttribute('name', section.name)
        xml_Section_text = doc.createTextNode(section.text)
        xml_Section.appendChild(xml_Section_text)
        xml_Text.appendChild(xml_Section)

    xml_IgnoredRegions = doc.createElement('IgnoredRegions')
    xml_SubmissionLabel.appendChild(xml_IgnoredRegions)

    for ignore_region in annotation_file.ignore_regions:
        xml_IgnoredRegion = doc.createElement('IgnoredRegion')
        xml_IgnoredRegion.setAttribute('len', str(ignore_region.end-ignore_region.start))
        xml_IgnoredRegion.setAttribute('name', ignore_region.name)
        xml_IgnoredRegion.setAttribute('section', ignore_region.section)
        xml_IgnoredRegion.setAttribute('start', str(ignore_region.start))
        xml_IgnoredRegions.appendChild(xml_IgnoredRegion)

    xml_Mentions = doc.createElement('Mentions')
    xml_SubmissionLabel.appendChild(xml_Mentions)

    for entity in entities:
        xml_Mention = doc.createElement('Mention')
        xml_Mention.setAttribute('id', 'M'+str(entity_id))
        entity_id += 1
        xml_Mention.setAttribute('len', span_to_len(entity))
        xml_Mention.setAttribute('section', entity.section)
        xml_Mention.setAttribute('start', span_to_start(entity))
        xml_Mention.setAttribute('type', 'OSE_Labeled_AE')
        xml_Mentions.appendChild(xml_Mention)

    with codecs.open(os.path.join(opt.predict, doc_name), 'w', 'UTF-8') as fp:
        doc.writexml(fp, addindent=' ' * 2, newl='\n', encoding='UTF-8')



def remove_entity_in_the_ignore_region(ignore_regions, entities, section_id):
    ret = []

    for entity in entities:
        remove_this_entity = False
        for ignore_region in ignore_regions:
            if ignore_region.section != section_id:
                continue
            for span in entity.spans:
                if is_overlapped(span[0], span[1], ignore_region.start, ignore_region.end):
                    remove_this_entity = True
                    break
            if remove_this_entity:
                break
        if not remove_this_entity:
            entity.section = section_id
            ret.append(entity)

    return ret

def translateResultsintoEntities(sentences, predict_results):
    pred_entities = []
    sent_num = len(predict_results)
    for idx in range(sent_num):

        predict_list = predict_results[idx]
        sentence = sentences[idx]

        entities = get_ner_BIOHD_1234(predict_list, False)

        # find span based on tkSpan
        for entity in entities:
            for tkSpan in entity.tkSpans:
                span = [sentence[tkSpan[0]]['start'], sentence[tkSpan[1]]['end']]
                entity.spans.append(span)


        pred_entities.extend(entities)


    return pred_entities

def test(data, opt):

    corpus_dir = opt.test_file

    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise RuntimeError("invalid nlp tool")

    corpus_files = [f for f in os.listdir(corpus_dir) if f.find('.xml') != -1]

    model = SeqModel(data, opt)
    if opt.test_in_cpu:
        model.load_state_dict(
            torch.load(os.path.join(opt.output, 'model.pkl'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.output, 'model.pkl')))

    makedir_and_clear(opt.predict)

    ct_success = 0
    ct_error = 0

    for fileName in corpus_files:
        try:
            start = time.time()
            document, annotation_file = processOneFile_fda(fileName, corpus_dir, nlp_tool, False, opt.types, opt.type_filter)
            pred_entities = []

            for section in document:

                data.test_texts = []
                data.test_Ids = []
                read_instance_from_one_document(section, data.word_alphabet, data.char_alphabet, data.label_alphabet,
                                                data.test_texts, data.test_Ids, data)

                _, _, _, _, _, pred_results, _ = evaluate(data, opt, model, 'test', False, opt.nbest)

                entities = translateResultsintoEntities(section.sentences, pred_results)

                # remove the entity in the ignore_region and fill section_id
                section_id = section.name[section.name.rfind('_')+1: ]
                entities = remove_entity_in_the_ignore_region(annotation_file.ignore_regions, entities, section_id)

                pred_entities.extend(entities)


            dump_results(fileName, pred_entities, opt, annotation_file)

            end = time.time()
            logging.info("process %s complete with %.2fs" % (fileName, end - start))

            ct_success += 1
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            ct_error += 1

    logging.info("test finished, total {}, error {}".format(ct_success + ct_error, ct_error))