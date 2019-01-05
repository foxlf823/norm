import umls
import metamap
from os import listdir
from os.path import isfile, join
import os
from my_utils import get_bioc_file
from data_structure import Entity,Document
import logging
from norm_utils import evaluate_for_ehr

type_we_care = set(['ADE','SSLIF', 'Indication'])

def parse_bioc(directory, fileName):
    document = Document()
    document.name = fileName[:fileName.find('.')]

    annotation_file = get_bioc_file(join(directory, fileName))
    bioc_passage = annotation_file[0].passages[0]
    entities = []

    for entity in bioc_passage.annotations:
        if entity.infons['type'] not in type_we_care:
            continue

        entity_ = Entity()
        entity_.id = entity.id
        entity_.name = entity.text
        entity_.type = entity.infons['type']
        entity_.spans.append([entity.locations[0].offset, entity.locations[0].end])

        if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A') \
                and ('SNOMED term' in entity.infons and entity.infons['SNOMED term'] != 'N/A'):
            entity_.norm_ids.append(entity.infons['SNOMED code'])
            entity_.norm_names.append(entity.infons['SNOMED term'])

        elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A') \
                and ('MedDRA term' in entity.infons and entity.infons['MedDRA term'] != 'N/A'):
            entity_.norm_ids.append(entity.infons['MedDRA code'])
            entity_.norm_names.append(entity.infons['MedDRA term'])
        else:
            logging.debug("{}: no norm id in entity {}".format(fileName, entity.id))
            # some entities may have no norm id

        entities.append(entity_)

    document.entities = entities

    return document


if __name__=="__main__":


    print("load umls ...")
    UMLS_dict, _ = umls.load_umls_MRCONSO("/Users/feili/UMLS/2016AA_Snomed_Meddra/META/MRCONSO.RRF")

    # base_dir =
    predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
    gold_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/bioc"

    ct_ner_predict = 0
    ct_ner_gold = 0
    ct_ner_correct = 0

    ct_norm_predict = 0
    ct_norm_gold = 0
    ct_norm_correct = 0


    for gold_file_name in listdir(gold_dir):

        gold_document = parse_bioc(gold_dir, gold_file_name)

        predict_document = metamap.load_metamap_result_from_file(join(predict_dir, gold_file_name[:gold_file_name.find('.')]+".field.txt"))

        ct_ner_gold += len(gold_document.entities)
        ct_ner_predict += len(predict_document.entities)

        for predict_entity in predict_document.entities:

            for gold_entity in gold_document.entities:

                if predict_entity.equals_span(gold_entity):

                    ct_ner_correct += 1

                    break



        p1, p2, p3 = evaluate_for_ehr(gold_document.entities, predict_document.entities, UMLS_dict)

        ct_norm_gold += p1
        ct_norm_predict += p2
        ct_norm_correct += p3





    p = ct_ner_correct*1.0/ct_ner_predict
    r = ct_ner_correct*1.0/ct_ner_gold
    f1 = 2.0*p*r/(p+r)
    print("NER p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

    p = ct_norm_correct*1.0/ct_norm_predict
    r = ct_norm_correct*1.0/ct_norm_gold
    f1 = 2.0*p*r/(p+r)
    print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

