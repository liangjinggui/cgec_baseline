#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import argparse
import pickle
from collections import Counter, defaultdict
from itertools import zip_longest


from seq2seq.utils import import_user_module
from seq2seq.data.dictionary import Dictionary
from seq2seq.binarizer import Binarizer
from seq2seq import tokenizer
from seq2seq import options


from multiprocessing import Pool

import os
import shutil

def main(args):
    import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    # data/train.source
    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname
    
    # 加上最终存储路径的前缀
    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    # data-bin/dict.source.txt
    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        # task.build_dictionary
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, args.workers)
        d.finalize(
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor
        )
        return d

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.copy_ext_dict:
        assert args.joined_dictionary, \
            "--joined-dictionary must be set if --copy-extended-dictionary is specified"
        assert args.workers == 1, \
            "--workers must be set to 1 if --copy-extended-dictionary is specified"

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = Dictionary.load(args.srcdict)
        elif args.tgtdict:
            src_dict = Dictionary.load(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            print('| Building dictionary ...')
            src_dict = build_dictionary(
                [train_path(lang) for lang in [args.source_lang, args.target_lang]], src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = Dictionary.load(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)
        
        if target:
            if args.tgtdict:
                tgt_dict = Dictionary.load(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    # 保存字典文件
    print("| Saving source dictionary to the {}".format(dict_path(args.source_lang)))
    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        print("| Saving target dictionary to the {}".format(dict_path(args.target_lang)))
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, copy_src_words=None):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()
        copyied = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            copyied.update(worker_result["copied"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]
        # data/train.source
        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        
        if num_workers > 1:  # todo: not support copy 
            raise argparse.ArgumentError("num_workers must be set to 1")
        # words_list:存放tokens的list
        # ids_list:存放tokens对应的id的list
        words_list = []
        ids_list = []
        def binarize_consumer(ids, words):
            ids_list.append(ids)
            words_list.append(words)
        # Binarizer.binarize 方法返回的时一个字典
        # 其中key包含：nseq、nunk、ntok、replaced、copied
        print("| Encoding the {} ...".format(input_file))
        merge_result(
            Binarizer.binarize(
                input_file, vocab, binarize_consumer,
                offset=0, end=offsets[1], copy_ext_dict=args.copy_ext_dict, copy_src_words=copy_src_words
            )
        )

        # data-bin/train.source-target.source.bin
        pickle.dump(ids_list, open(dataset_dest_file(args, output_prefix, lang, "bin"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}, {:.3}% <unk> copied from src".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
                100 * sum(copyied.values()) / n_seq_tok[1]
            )
        )

        return words_list

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, copy_src_words=None):
        # vocab: dictionary类，已有字典
        # input_prefix: 训练或测试数据前缀
        # output_prefix: 输出文件路径前缀
        # lang：训练数据类型后缀
        # num_workers: cgec任务中只能设为1
        # copy_src_words: 处理src数据时为None，处理tgt数据时需要特别设置
        if args.output_format == "binary":
            return make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, copy_src_words)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            # data-bin/test.source-target.source
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
            return None

    # 遍历语料库，将token根据字典映射成id并保存
    def make_all(lang, vocab, source_words_list_dict=defaultdict(lambda: None)):
        words_list_dict = defaultdict(lambda: None)

        if args.trainpref:
            words_list_dict["train"] = \
                make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers,
                             copy_src_words=source_words_list_dict['train'])
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                words_list_dict["valid"] = \
                    make_dataset(vocab, validpref, outprefix, lang, copy_src_words=source_words_list_dict['valid'])
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                words_list_dict["test"] = \
                    make_dataset(vocab, testpref, outprefix, lang, copy_src_words=source_words_list_dict['test'])

        return words_list_dict
    
    # 处理source语料时，copy_src_words为None，不起作用，不用copy
    source_words_list_dict = make_all(args.source_lang, src_dict)
    # 处理target数据时，需要输入src_words_list_dict作为copy字典
    if target:
        target_words_list_dict = make_all(args.target_lang, tgt_dict, source_words_list_dict)

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if False:
        # TODO 实现cgec position embedding方法
        pass


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)




if __name__ == '__main__':
    
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)