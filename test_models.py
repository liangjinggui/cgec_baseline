from seq2seq.data.dictionary import Dictionary

from seq2seq.models.transformer import TransformerModel
from seq2seq import options
import argparse
if __name__ == '__main__':
    path = './out/data_bin/dict.source.txt'
    d = Dictionary.load(path)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    model = TransformerModel.build_model(args, d, d)
    
    print(model)
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))