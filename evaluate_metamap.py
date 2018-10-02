import umls
import metamap
from os import listdir
from os.path import isfile, join
import os
from my_utils import get_bioc_file
from data_structure import Entity,Document

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


        if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A'):
            id = entity.infons['SNOMED code']
        elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A'):
            id = entity.infons['MedDRA code']
        else:
            print("annotation error, ignored: {} in {}".format(entity.id, fileName))
            continue

        entity_ = Entity()
        if isinstance(entity.text, str):
            text = entity.text.decode('utf-8')
        else:  # unicode
            text = entity.text
        entity_.create(entity.id, entity.infons['type'], entity.locations[0].offset, entity.locations[0].end,
                       text, None, None, None)
        entity_.norm_id = id

        entities.append(entity_)

    document.entities = entities

    return document


if __name__=="__main__":


    print("load umls ...")
    UMLS_dict = umls.load_umls_MRCONSO("/Users/feili/UMLS/2016AA/META/MRCONSO.RRF")

    # base_dir =
    predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
    gold_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/bioc"

    ct_predict = 0
    ct_gold = 0
    ct_norm_correct = 0

    ct_ner_correct = 0


    for gold_file_name in listdir(gold_dir):

        gold_document = parse_bioc(gold_dir, gold_file_name)

        predict_document = metamap.load_metamap_result_from_file(join(predict_dir, gold_file_name[:gold_file_name.find('.')]+".field.txt"))

        ct_gold += len(gold_document.entities)
        ct_predict += len(predict_document.entities)

        for predict_entity in predict_document.entities:

            for gold_entity in gold_document.entities:

                if predict_entity.equals_span(gold_entity):

                    ct_ner_correct += 1

                    if predict_entity.norm_id in UMLS_dict:
                        concept = UMLS_dict[predict_entity.norm_id]

                        if gold_entity.norm_id in concept.codes:
                            ct_norm_correct += 1

                    break






    p = ct_ner_correct*1.0/ct_predict
    r = ct_ner_correct*1.0/ct_gold
    f1 = 2.0*p*r/(p+r)
    print("NER p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

    p = ct_norm_correct*1.0/ct_predict
    r = ct_norm_correct*1.0/ct_gold
    f1 = 2.0*p*r/(p+r)
    print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

