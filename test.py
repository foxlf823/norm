from seqmodel import SeqModel
import torch
import os
from my_utils import  evaluate
import codecs


def dump_results(sentences, predict_results, file_path):
    if len(sentences) != len(predict_results):
        raise RuntimeError("sentence number not match")
    with codecs.open(file_path, 'w', 'UTF-8') as f:
        for i, sentence in enumerate(sentences):
            if len(sentence) != len(predict_results[i][0]):
                raise RuntimeError("token number not match")

            predict = predict_results[i][0]

            for j, token in enumerate(sentence):

                line = token['text']+" "+token['doc']+" "+unicode(token['start'])+" "\
                       +unicode(token['end'])+" "+predict[j]+"\n"
                f.write(line)

            f.write("\n")



def test(data, opt):
    model = SeqModel(data, opt)

    model.load_state_dict(torch.load(os.path.join(opt.output, 'model.pkl')))

    _, _, p, r, f, pred_results, _ = evaluate(data, opt, model, 'test', opt.nbest)

    dump_results(data.test_data, pred_results, os.path.join(opt.output, 'result.txt'))

    print("test finished")