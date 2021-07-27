#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    parser.add_argument('--data_dir', type=str, default="../../data-files/wikitext-15M/",
                        help='Whether to resume for this file.')
    parser.add_argument('--output_dir', type=str, default="../../data-files/wikitext-15M-conllu/",
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
        args.data_dir="../../data-files/wikitext-15M/"
        args.output_dir="../../data-files/wikitext-15M-conllu/"
        args.seed=42
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
    wiki_datasets = DatasetDict.load_from_disk(args.data_dir)
    
    logging.info("Removing any existing files including:")
    # output file cleanup if exist.
    train_output_file = os.path.join(args.output_dir, "wikitext-15M-train.conllu")
    test_output_file = os.path.join(args.output_dir, "wikitext-15M-test.conllu")
    validation_output_file = os.path.join(args.output_dir, "wikitext-15M-validation.conllu")
    logging.info(train_output_file)
    logging.info(validation_output_file)
    logging.info(test_output_file)
    try:
        os.remove(train_output_file)
        os.remove(test_output_file)
        os.remove(validation_output_file)
    except OSError:
        pass
    
    # we are appending so later scripts can process this file in batch?
    write_mode = "a+"
    
    # Stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_no_ssplit=True)
    logging.info("Finish loading Stanza in.")
    
    def preprocess(wiki_datasets, args, split):
        
        logging.info(f"Preprocessing split={split}.")
        sentences = []
        count = 0
        for s in wiki_datasets[split]:
            if len(s["text"].strip()) > 0:
                clean_s = []
                for t in s["text"].strip().split(" "):
                    if len(t.strip()) > 0:
                        clean_s += [t.strip()]
                sentences += [" ".join(clean_s)] # we strip it, and split by space.
                count += 1
                if count == args.max_number_of_examples:
                    break
        
        chunks = list(partition(sentences, args.batch_size))
        total_chunk = len(chunks)
        count = 0
        
        if split == "train":
            output_file = train_output_file
        elif split == "test":
            output_file = test_output_file
        else:
            output_file = validation_output_file
        
        for chunk in chunks:
            logging.info(f"processing: {count+1}/{total_chunk}.")
            in_docs = [stanza.Document([], text=d) for d in chunk]
            docs = nlp(in_docs)
            for i in range(len(docs)):
                CoNLL.write_doc2conll(docs[i], output_file, mode=write_mode)
            count += 1
            
    if args.include_train:
        preprocess(wiki_datasets, args, "train")
    if args.include_validation:
        preprocess(wiki_datasets, args, "validation")
    if args.include_test:
        preprocess(wiki_datasets, args, "test")
    
    logging.info("Saved Pos-tagging with data to:")
    logging.info(args.output_dir)
    
    logging.info(f"Finish.")

