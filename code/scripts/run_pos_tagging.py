#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import stanza
import os
import argparse
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset
from datasets import list_datasets
import logging
import pathlib


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
    parser.add_argument('--output_dir', type=str, default="../../data-files/wikitext-15M-pos/",
                        help='Whether to resume for this file.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
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
        args.output_dir="../../data-files/wikitext-15M-pos/"
        args.seed=42
        is_jupyter = True
    except:
        is_jupyter = False
    
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
    
    logging.info("Running Pos-tagging with data lives in:")
    logging.info(args.data_dir)
    wiki_datasets = DatasetDict.load_from_disk(args.data_dir)
    
    # Stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
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
                        clean_s += [t]
                sentences += [clean_s] # we strip it, and split by space.
                count += 1
                if count == args.max_number_of_examples:
                    break
        
        chunks = list(partition(sentences, args.batch_size))
        total_chunk = len(chunks)
        count = 0
        
        sentence_strs = []
        upos_strs = []
        xpos_strs = []
        
        for chunk in chunks:
            logging.info(f"processing: {count+1}/{total_chunk}.")
            doc = nlp(chunk)
            for i, sentence in enumerate(doc.sentences):
                sentence_str = []
                upos_str = []
                xpos_str = []
                for token in sentence.tokens:
                    sentence_str += [token.text]
                    upos_str += [token.words[0].upos]
                    xpos_str += [token.words[0].xpos]
                sentence_strs += [sentence_str]
                upos_strs += [upos_str]
                xpos_strs += [xpos_str]
            count += 1
        examples = {
            "sentence_str":sentence_strs,
            "upos_str":upos_strs,
            "xpos_str":xpos_strs,
        }
        return examples
    
    examples_train = Dataset.from_dict(preprocess(wiki_datasets, args, "train"))
    examples_validation = Dataset.from_dict(preprocess(wiki_datasets, args, "validation"))
    examples_test = Dataset.from_dict(preprocess(wiki_datasets, args, "test"))
    
    dataset = DatasetDict({
        "train": examples_train,
        "validation": examples_validation,
        "test": examples_test,
    })
    logging.info("Saving Pos-tagging with data to:")
    logging.info(args.output_dir)
    dataset.save_to_disk(args.output_dir)
    total_count = len(examples_train) + len(examples_validation) + len(examples_test)
    logging.info(f"Collected in total {total_count} examples.")

