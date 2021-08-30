#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
UPDATES:
When we read in long paragraphs which contain multiple sentences,
we will need Stanza to chunk them into sentences. This causes some
problems when generating conullu files where a "sentence" will be
essentially chunked into multiple sentences, and we need to merge
sentences back when generating the perturbed datasets.
"""


# In[9]:


# Imports
import stanza
from stanza.utils.conll import CoNLL
import os
import argparse
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset
from datasets import list_datasets
import logging
import pathlib
import random
import torch
import numpy as np
import json
import copy
import pandas as pd

text_fields_map = {
    "wikitext-15M":"text",
    "sst3":"text",
    "qnli":"question,sentence",
    "mrpc":"sentence1,sentence2",
}


# In[ ]:


# Utils
def get_sentence_doc(sentence_in):
    doc = nlp(sentence_in)
    return doc

def get_postag_token(sentence_in):
    ret = []
    doc = nlp(sentence_in)
    for sent in doc.sentences:
        for word in sent.words:
            ret  += [(word.text, word.upos, word.xpos,)]
    return ret

def partition(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


def arg_parse():
    
    parser = argparse.ArgumentParser(description='pos-tagging config.')
    # Experiment management:

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size.')
    parser.add_argument('--max_number_of_examples', type=int, default=-1,
                        help='Max number of examples to load for each splits.')
    parser.add_argument('--data_dir', type=str, default="../../data-files/",
                        help='Whether to resume for this file.')
    parser.add_argument('--task', type=str, default="sst3",
                        help='Whether to resume for this file.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument("--include_train",
                       default=False,
                       action='store_true',
                       help="Whether to include train.")  
    parser.add_argument("--include_test",
                       default=False,
                       action='store_true',
                       help="Whether to include test.")  
    parser.add_argument("--include_validation",
                       default=False,
                       action='store_true',
                       help="Whether to run eval on the test set.")  
    parser.add_argument("--include_all_feilds",
                       default=True,
                       action='store_true',
                       help="Whether to include all fields.")  
    parser.set_defaults(
        # Exp management:
        seed=42,
    )
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args


# In[ ]:


if __name__ == "__main__":
    
    # Loading arguments
    args = arg_parse()
    try:        
        get_ipython().run_line_magic('matplotlib', 'inline')
        # Experiment management:
        args.batch_size=128
        args.data_dir="../../data-files/"
        args.task="sst3"
        args.seed=42
        include_all_feilds = True
        is_jupyter = True
    except:
        is_jupyter = False
        
    if not args.include_train and not args.include_test and not args.include_validation:
        logging.error("Need to at least specify a single partition.")
        assert False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
    # Create output directory if not exists.
    args.output_dir = os.path.join(args.data_dir, f"{args.task}-conllu")
    args.data_dir = os.path.join(args.data_dir, args.task)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)-8s %(message)s', 
        datefmt='%a, %d %b %Y %H:%M:%S', 
        filename=os.path.join(args.output_dir, "training.log"),
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler(os.sys.stdout))
    
    logging.info("Args:")
    logging.info(args)
    
    logging.info("Running conllu transformation with data lives in:")
    logging.info(args.data_dir)
    datasets = DatasetDict.load_from_disk(args.data_dir)
    
    if args.task not in text_fields_map:
        logging.error("You task is not supported by this script: ", args.task)
        
    text_field = text_fields_map[args.task].split(",")
    if len(text_field) > 1:
        logging.info("This dataset contains multiple text fields to shift: ", text_field)
    
    # collecing all other fields besides the text field.
    other_fields = []
    for f in datasets["train"][0].keys():
        if f not in text_field:
            other_fields += [f]
    logging.info("You are also including these metadata in your data: ", other_fields)
    
    logging.info("Removing any existing files including:")
    # output file cleanup if exist.
    train_output_file = os.path.join(args.output_dir, f"{args.task}-train")
    test_output_file = os.path.join(args.output_dir, f"{args.task}-test")
    validation_output_file = os.path.join(args.output_dir, f"{args.task}-validation")
    train_json_file = os.path.join(args.output_dir, f"{args.task}-train.json")
    test_json_file = os.path.join(args.output_dir, f"{args.task}-test.json")
    validation_json_file = os.path.join(args.output_dir, f"{args.task}-validation.json")
    logging.info(train_output_file)
    logging.info(validation_output_file)
    logging.info(test_output_file)
    logging.info(train_json_file)
    logging.info(test_json_file)
    logging.info(validation_json_file)
    try:
        for text_f in text_field:
            os.remove(train_output_file+f"-{text_f}.conllu")
            os.remove(test_output_file+f"-{text_f}.conllu")
            os.remove(validation_output_file+f"-{text_f}.conllu")
        os.remove(train_json_file)
        os.remove(test_json_file)
        os.remove(validation_json_file)
    except OSError:
        pass
    
    # we are appending so later scripts can process this file in batch?
    write_mode = "a+"
    
    # Stanza
    # We need to set tokenize_no_ssplit to False, and we need to merge at later stage
    # so that our shifting is NOT across sentences which would not make sense for long
    # sequences in the wiki-text data, for instance.
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_no_ssplit=False)
    logging.info("Finish loading Stanza in.")
    
    def preprocess(datasets, args, split):
        
        logging.info(f"Preprocessing split={split}.")
        sentences = []
        fields = []
        count = 0
        for s in datasets[split]:
            clean_ss = []
            for text in text_field:
                if s[text] != None and len(s[text].strip()) > 0:
                    clean_s = []
                    for t in s[text].strip().split(" "):
                        if len(t.strip()) > 0:
                            clean_s += [t.strip()]
                    clean_ss += [" ".join(clean_s)]
                else:
                    clean_ss += [""]
                    
            # we only allow where all fields are valid.
            if "" not in clean_ss:
                sentences += [dict(zip(text_field, clean_ss))]
                _fields = []
                for f in other_fields:
                    _fields += [s[f]]
                fields += [_fields]
                count += 1
                if count == args.max_number_of_examples:
                    break
        
        assert len(sentences) == len(fields), f"sentence count {len(sentences)} is not equal to fields count {len(fields)}"
        
        chunks = list(partition(sentences, args.batch_size))
        total_chunk = len(chunks)
        count = 0
        
        if split == "train":
            output_file = train_output_file
            output_json = train_json_file
        elif split == "test":
            output_file = test_output_file
            output_json = test_json_file
        else:
            output_file = validation_output_file
            output_json = validation_json_file
        
        all_meta = {}
        for text_f in text_field:
            all_meta[text_f] = []
            # this is for corner case where different fields will have
            # different metadata for example conllu object counts.
            
        idx = 0
        for chunk in chunks:
            logging.info(f"processing: {count+1}/{total_chunk}.")
            # we need to also take care the multi text fields cases
            for text_f in text_field:
                in_docs = [stanza.Document([], text=d[text_f]) for d in chunk]
                docs = nlp(in_docs)
                for i in range(len(docs)):
                    CoNLL.write_doc2conll(docs[i], output_file+f"-{text_f}.conllu", mode=write_mode)
                    # count the number of sentences, and we need to save it somewhere
                    # so that we can merge them back at the end.
                    sentence_count = len(docs[i].sentences)
                    meta = copy.deepcopy(fields[idx+i])
                    meta += [sentence_count]
                    all_meta[text_f] += [meta]

            idx += len(chunk)
            count += 1
        logging.info("Saving sentence slicing information and metadata (other fields) into:")
        logging.info(output_json)
        # dump to the disk.
        with open(output_json, "w") as fd:
            json.dump(all_meta, fd, indent=4)
            
    if args.include_train:
        preprocess(datasets, args, "train")
    if args.include_validation:
        preprocess(datasets, args, "validation")
    if args.include_test:
        preprocess(datasets, args, "test")
    
    logging.info("Saved Conllu files with metadata to:")
    logging.info(args.output_dir)
    
    logging.info(f"Finish.")

