#!/usr/bin/env python
# coding: utf-8

# In[80]:


# Load modules, mainly huggingface basic model handlers.
# Make sure you install huggingface and other packages properly.
from collections import Counter
import json
import copy
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef
from vocab_mismatch_utils import *

import logging
logger = logging.getLogger(__name__)

import os
os.environ["TRANSFORMERS_CACHE"] = "../huggingface_cache/" # Not overload common dir 
                                                           # if run in shared resources.

import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import DatasetDict

import transformers
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from functools import partial


# In[81]:


def generate_training_args(args, perturbed_type):
    
    training_args = TrainingArguments("tmp_trainer")
    training_args.no_cuda = args.no_cuda
    training_args.seed = args.seed
    training_args.do_train = args.do_train
    training_args.do_eval = args.do_eval

    training_args.evaluation_strategy = args.evaluation_strategy # evaluation is done after each epoch
    training_args.metric_for_best_model = args.metric_for_best_model
    training_args.greater_is_better = args.greater_is_better
    training_args.logging_dir = args.logging_dir
    training_args.task_name = args.task_name
    training_args.learning_rate = args.learning_rate
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    training_args.num_train_epochs = args.num_train_epochs # this is the maximum num_train_epochs, we set this to be 100.
    training_args.eval_steps = args.eval_steps
    training_args.logging_steps = args.logging_steps
    training_args.load_best_model_at_end = args.load_best_model_at_end
    if args.save_total_limit != -1:
        # only set if it is specified
        training_args.save_total_limit = args.save_total_limit

    training_args.log_level = "info"
    training_args.log_level_replica = "info"
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    
    logger.info(f"Recieved raw args: {args}")
    
    # validation of inputs.
    if "/" not in args.model_name_or_path:
        logger.warning("WARNING: you have to use your saved model from mid-tuning, not brand-new models.")
    if "/" not in args.tokenizer_name:
        logger.warning("WARNING: you have to use your saved tokenizer from mid-tuning, not brand-new tokenizers.")
        
    logger.info("Generating the run name for WANDB for better experiment tracking.")
    import datetime
    date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)
    
    if len(args.model_name_or_path.split("/")) > 1:
        run_name = "{0}_task_{1}_ft_{2}".format(
            date_time,
            args.task_name,
            "_".join(args.model_name_or_path.split("/")[1].split("_")[1:]),
        )
    else:
        if args.no_pretrain:
            run_name = "{0}_task_{1}_ft_{2}_no_pretrain_reinit_emb_{3}_reinit_avg_{4}_token_s_{5}_word_s_{6}_lr_{7}_seed_{8}_reverse_{9}_random_{10}".format(
                date_time,
                args.task_name,
                args.model_name_or_path,
                args.reinit_embeddings,
                args.reinit_avg_embeddings,
                args.token_swapping,
                args.word_swapping,
                args.learning_rate,
                args.seed,
                args.reverse_order,
                args.random_order,
            )
        else:
            run_name = "{0}_task_{1}_ft_{2}_reinit_emb_{3}_reinit_avg_{4}_token_s_{5}_word_s_{6}_lr_{7}_seed_{8}_reverse_{9}_random_{10}".format(
                date_time,
                args.task_name,
                args.model_name_or_path,
                args.reinit_embeddings,
                args.reinit_avg_embeddings,
                args.token_swapping,
                args.word_swapping,
                args.learning_rate,
                args.seed,
                args.reverse_order,
                args.random_order,
            )
    training_args.run_name = run_name
    logger.info(f"WANDB RUN NAME: {training_args.run_name}")
    training_args.output_dir = os.path.join(args.output_dir, run_name)

    training_args_dict = training_args.to_dict()
    # for PR
    _n_gpu = training_args_dict["_n_gpu"]
    del training_args_dict["_n_gpu"]
    training_args_dict["n_gpu"] = _n_gpu
    HfParser = HfArgumentParser((TrainingArguments))
    training_args = HfParser.parse_dict(training_args_dict)[0]

    if args.model_name_or_path == "":
        assert False # you have to provide one of them.
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    return training_args

def random_corrupt(task, tokenizer, vocab_match, example):
    # for tasks that have single sentence
    if task == "sst3" or task == "wiki-text" or task == "cola":
        original_sentence = example[TASK_CONFIG[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][0]] = corrupted_sentence
    # for tasks that have two sentences
    elif task == "mrpc" or task == "mnli" or task == "snli" or task == "qnli":
        original_sentence = example[TASK_CONFIG[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][0]] = corrupted_sentence
        
        original_sentence = example[TASK_CONFIG[task][1]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][1]] = corrupted_sentence
    elif task == "conll2003" or task == "en_ewt":
        original_tokens = example[TASK_CONFIG[task][0]]
        corrupted_tokens = [vocab_match[t] for t in original_tokens]
        example[TASK_CONFIG[task][0]] = corrupted_tokens
    else:
        print(f"task={task} not supported yet!")
    return example


# In[ ]:


class HuggingFaceRoBERTaBase:
    """
    An extension for evaluation based off the huggingface module.
    """
    def __init__(self, tokenizer, model, task_name, task_config):
        self.task_name = task_name
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.model = model
        
    def train(self, inoculation_train_df, eval_df, model, args, training_args, max_length=128,
              inoculation_patience_count=-1, pd_format=True, 
              scramble_proportion=0.0, eval_with_scramble=False):

        if pd_format:
            datasets = {}
            datasets["train"] = Dataset.from_pandas(inoculation_train_df)
            datasets["validation"] = Dataset.from_pandas(eval_df)
        else:
            datasets = {}
            datasets["train"] = inoculation_train_df
            datasets["validation"] = eval_df
        logger.info(f"***** Train Sample Count (Verify): %s *****"%(len(datasets["train"])))
        logger.info(f"***** Valid Sample Count (Verify): %s *****"%(len(datasets["validation"])))
    
        label_list = datasets["validation"].unique("label")
        label_list.sort()  # Let's sort it for determinism

        sentence1_key, sentence2_key = self.task_config
        padding = "max_length"
        label_to_id = None
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[l] for l in examples["label"]]
            return result
        datasets["train"] = datasets["train"].map(preprocess_function, batched=True)
        datasets["validation"] = datasets["validation"].map(preprocess_function, batched=True)
        
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]
        
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            
        metric = load_metric("glue", "sst2") # any glue task will do the job, just for eval loss
        
        def asenti_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result_to_print = classification_report(p.label_ids, preds, digits=5, output_dict=True)
            print(classification_report(p.label_ids, preds, digits=5))
            mcc_scores = matthews_corrcoef(p.label_ids, preds)
            logger.info(f"MCC scores: {mcc_scores}.")
            result_to_return = metric.compute(predictions=preds, references=p.label_ids)
            result_to_return["Macro-F1"] = result_to_print["macro avg"]["f1-score"]
            result_to_return["MCC"] = mcc_scores
            return result_to_return

        # Initialize our Trainer. We are only intersted in evaluations
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=asenti_compute_metrics,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator
        )
        # Early stop
        if inoculation_patience_count != -1:
            trainer.add_callback(EarlyStoppingCallback(inoculation_patience_count))
        
        # Training
        if training_args.do_train:
            logger.info("*** Training our model ***")
            trainer.train()
            trainer.save_model()  # Saves the tokenizer too for easy upload
        
        # Evaluation
        eval_results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            tasks = [self.task_name]
            eval_datasets = [eval_dataset]
            for eval_dataset, task in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in eval_result.items():
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")
                eval_results.update(eval_result)


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--wandb_proj_name",
                        default="",
                        type=str)
    parser.add_argument("--task_name",
                        default="sst3",
                        type=str)
    parser.add_argument("--train_file",
                        default="../data-files/sst/sst-tenary-train.tsv",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_file",
                        default="../data-files/sst-tenary/sst-tenary-dev.tsv",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path",
                        default="../7-31_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0/",
                        type=str,
                        help="The pretrained model binary file.")
    parser.add_argument("--tokenizer_name",
                        default="../7-31_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0/",
                        type=str,
                        help="Tokenizer name.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--evaluation_strategy",
                        default="steps",
                        type=str,
                        help="When you evaluate your training model on eval set.")
    parser.add_argument("--cache_dir",
                        default="../tmp/",
                        type=str,
                        help="Cache directory for the evaluation pipeline (not HF cache).")
    parser.add_argument("--logging_dir",
                        default="../tmp/",
                        type=str,
                        help="Logging directory.")
    parser.add_argument("--output_dir",
                        default="../",
                        type=str,
                        help="Output directory of this training process.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--metric_for_best_model",
                        default="Macro-F1",
                        type=str,
                        help="The metric to use to compare two different models.")
    parser.add_argument("--greater_is_better",
                        default=True,
                        action='store_true',
                        help="Whether the `metric_for_best_model` should be maximized or not.")
    parser.add_argument("--is_tensorboard",
                        default=False,
                        action='store_true',
                        help="If tensorboard is connected.")
    parser.add_argument("--load_best_model_at_end",
                        default=True,
                        action='store_true',
                        help="Whether load best model and evaluate at the end.")
    parser.add_argument("--eval_steps",
                        default=10,
                        type=float,
                        help="The total steps to flush logs to wandb specifically.")
    parser.add_argument("--logging_steps",
                        default=10,
                        type=float,
                        help="The total steps to flush logs to wandb specifically.")
    parser.add_argument("--save_total_limit",
                        default=1,
                        type=int,
                        help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output dir.")
    # these are arguments for inoculations
    parser.add_argument("--inoculation_patience_count",
                        default=-1,
                        type=int,
                        help="If the evaluation metrics is not increasing with maximum this step number, the training will be stopped.")
    parser.add_argument("--per_device_train_batch_size",
                        default=64,
                        type=int,
                        help="")
    parser.add_argument("--per_device_eval_batch_size",
                        default=64,
                        type=int,
                        help="")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="The total number of epochs for training.")
    parser.add_argument("--no_pretrain",
                        default=False,
                        action='store_true',
                        help="Whether to use pretrained model if provided.")
    parser.add_argument("--reinit_avg_embeddings",
                        default=False,
                        action='store_true',
                        help="Whether to reinit embeddings to be the average embeddings.")
    parser.add_argument("--reinit_embeddings",
                        default=False,
                        action='store_true',
                        help="Whether to reinit embeddings to be the random embeddings.")
    parser.add_argument("--token_swapping",
                        default=False,
                        action='store_true',
                        help="Whether to swap token randomly.")
    parser.add_argument("--word_swapping",
                        default=False,
                        action='store_true',
                        help="Whether to swap words randomly.")
    parser.add_argument("--swap_vocab_file",
                        default="../data-files/wikitext-15M-vocab.json",
                        type=str,
                        help="Please provide a vocab file if you want to do word swapping.")
    parser.add_argument("--train_embeddings_only",
                        default=False,
                        action='store_true',
                        help="If only train embeddings not the whole model.")
    parser.add_argument("--train_linear_layer_only",
                        default=False,
                        action='store_true',
                        help="If only train embeddings not the whole model.")
    # these are arguments for scrambling texts
    parser.add_argument("--reverse_order",
                        default=False,
                        action='store_true',
                        help="Whether to reverse the sequence order.")
    parser.add_argument("--random_order",
                        default=False,
                        action='store_true',
                        help="Whether to random order the sequence.")
    parser.add_argument("--scramble_proportion",
                        default=0.0,
                        type=float,
                        help="What is the percentage of text you want to scramble.")
    parser.add_argument("--inoculation_p",
                        default=1.0,
                        type=float,
                        help="How many data you need to train")
    parser.add_argument("--eval_with_scramble",
                        default=False,
                        action='store_true',
                        help="If you are also evaluating with scrambled texts.")
    parser.add_argument("--n_layer_to_finetune",
                        default=-1,
                        type=int,
                        help="Indicate a number that is less than original layer if you only want to finetune with earlier layers only.")

    args, unknown = parser.parse_known_args()
    # os.environ["WANDB_DISABLED"] = "NO" if args.is_tensorboard else "YES" # BUG
    os.environ["TRANSFORMERS_CACHE"] = "../huggingface_inoculation_cache/"
    os.environ["WANDB_PROJECT"] = f"fine_tuning"
    
    # if cache does not exist, create one
    if not os.path.exists(os.environ["TRANSFORMERS_CACHE"]): 
        os.makedirs(os.environ["TRANSFORMERS_CACHE"])
    TASK_CONFIG = {
        "sst3": ("text", None),
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "snli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence")
    }
    # Load pretrained model and tokenizer
    NUM_LABELS = 2 if args.task_name == "cola" or args.task_name == "mrpc" or args.task_name == "qnli" else 3
    MAX_SEQ_LEN = args.max_seq_length
    
    args.tokenizer_name = args.model_name_or_path
    name_list = args.model_name_or_path.split("_")
    
    perturbed_type = ""
    
    inoculation_p = 0.0
    if not args.no_pretrain:
        for i in range(len(name_list)):
            if name_list[i] == "seed":
                args.seed = int(name_list[i+1])
            if name_list[i] == "reverse":
                if name_list[i+1] == "True":
                    args.reverse_order = True
                else:
                    args.reverse_order = False
            if name_list[i] == "random":
                if name_list[i+1].strip("/") == "True":
                    args.random_order = True
                else:
                    args.random_order = False
            if name_list[i] == "data":
                if len(name_list[i+1].split("-")) > 2:
                    perturbed_type = "-".join(name_list[i+1].split("-")[2:])
            if name_list[i] == "inoculation":
                inoculation_p = float(name_list[i+1])
    
    if perturbed_type == "":
        args.train_file = f"../data-files/{args.task_name}"
    else:
        args.train_file = f"../data-files/{args.task_name}-{perturbed_type}"
    
    training_args = generate_training_args(args, perturbed_type)
    
    need_resize = False
    if inoculation_p == 0.0:
        logger.warning(f"***** WARNING: Detected inoculation_p={inoculation_p}; initialize the model and the tokenizer from huggingface. *****")
        # we need to make sure tokenizer is the correct one!
        if "albert-base-v2" in args.model_name_or_path:
            args.tokenizer_name = "albert-base-v2"
            need_resize = True
        elif "bert-base-cased" in args.model_name_or_path:
            args.tokenizer_name = "bert-base-cased"
            need_resize = True
        else:
            args.tokenizer_name = "roberta-base"
        args.model_name_or_path = "roberta-base"
        
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=NUM_LABELS,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir
    )
    
    if need_resize:
        # we need to rewrite the number of type token a little
        # during pretraining, there are two types for reberta
        # during fine-tuning, i think we are only using one?
        if args.tokenizer_name == "albert-base-v2":
            config.type_vocab_size = 1
        else:
            config.type_vocab_size = 1

    if args.n_layer_to_finetune != -1:
        # then we are only finetuning n-th layer, not all the layers
        if args.n_layer_to_finetune > config.num_hidden_layers:
            logger.info(f"***** WARNING: You are trying to train with first {args.n_layer_to_finetune} layers only *****")
            logger.info(f"***** WARNING: But the model has only {config.num_hidden_layers} layers *****")
            logger.info(f"***** WARNING: Training with all layers instead! *****")
            pass # just to let it happen, just train it with all layers
        else:
            # overwrite
            logger.info(f"***** WARNING: You are trying to train with first {args.n_layer_to_finetune} layers only *****")
            logger.info(f"***** WARNING: But the model has only {config.num_hidden_layers} layers *****")
            config.num_hidden_layers = args.n_layer_to_finetune

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=False,
        cache_dir=args.cache_dir
    )
    
    if args.no_pretrain:
        logger.info("***** Training new model from scratch *****")
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir
        )

    if need_resize:
        logger.info("***** Replacing the word_embeddings and token_type_embeddings with random initialized values *****")
        # this means, we are finetuning directly with new tokenizer.
        # so the model itself has a different tokenizer, we need to resize.
        model.resize_token_embeddings(len(tokenizer))
        # If we resize, we also enforce it to reinit
        # so we are controlling for weights distribution.
        random_config = AutoConfig.from_pretrained(
            args.model_name_or_path, 
            num_labels=NUM_LABELS,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir
        )
        # we need to check if type embedding need to be resized as well.
        tokenizer_config = AutoConfig.from_pretrained(
            args.tokenizer_name, 
            num_labels=NUM_LABELS,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir
        )
        # IMPORTANT: THIS ENSURES TYPE WILL NOT CAUSE UNREF POINTER ISSUE.
        random_config.type_vocab_size = tokenizer_config.type_vocab_size
        random_model = AutoModelForSequenceClassification.from_config(
            config=random_config,
        )
        random_model.resize_token_embeddings(len(tokenizer))
        replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data.clone()
        model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    
    if args.reinit_avg_embeddings:
        logger.info("***** WARNING: We reinit all embeddings to be the average embedding from the pretrained model. *****")
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir
        )
        avg_embeddings = torch.mean(pretrained_model.roberta.embeddings.word_embeddings.weight.data, dim=0).expand_as(model.roberta.embeddings.word_embeddings.weight.data)
        model.roberta.embeddings.word_embeddings.weight.data = avg_embeddings
        # to keep consistent, we also need to reinit the type embeddings.
        random_model = AutoModelForSequenceClassification.from_config(
            config=config,
        )
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    elif args.reinit_embeddings:
        logger.info("***** WARNING: We reinit all embeddings to be the randomly initialized embeddings. *****")
        random_model = AutoModelForSequenceClassification.from_config(config)
        # random_model.resize_token_embeddings(len(tokenizer))
        replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data.clone()
        model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        # to keep consistent, we also need to reinit the type embeddings.
        random_model = AutoModelForSequenceClassification.from_config(
            config=config,
        )
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    else:
        pass
    
    if args.token_swapping:
        logger.info("***** WARNING: We are swapping tokens via embeddings. *****")
        original_embeddings = model.roberta.embeddings.word_embeddings.weight.data.clone()
        g = torch.Generator()
        g.manual_seed(args.seed)
        perm_idx = torch.randperm(original_embeddings.size()[0], generator=g)
        swapped_embeddings = original_embeddings.index_select(dim=0, index=perm_idx)
        model.roberta.embeddings.word_embeddings.weight.data = swapped_embeddings
    elif args.word_swapping:
        logger.info("***** WARNING: We are swapping words in the inputs. *****")
        token_frequency_map = json.load(open(args.swap_vocab_file))
        wikitext_vocab = list(set(token_frequency_map.keys()))
        wikitext_vocab_copy = copy.deepcopy(wikitext_vocab)
        random.Random(args.seed).shuffle(wikitext_vocab_copy)
        word_swap_map = {}
        for i in range(len(wikitext_vocab)):
            word_swap_map[wikitext_vocab[i]] = wikitext_vocab_copy[i]
    
    assert len(tokenizer) == model.roberta.embeddings.word_embeddings.weight.data.shape[0]
    
    logger.info(f"***** Current setups *****")
    logger.info(f"***** model type: {args.model_name_or_path} *****")
    logger.info(f"***** tokenizer type: {args.tokenizer_name} *****")
    
    # We cannot resize this. In the mid-tuning, this is already resized.
    # if args.tokenizer_name != args.model_name_or_path:
    #     model.resize_token_embeddings(len(tokenizer))
        
    if args.train_embeddings_only:
        logger.info("***** We only train embeddings, not other layers *****")
        for name, param in model.named_parameters():
            if 'word_embeddings' not in name: # only word embeddings
                param.requires_grad = False
    
    if args.train_linear_layer_only:
        logger.info("***** We only train classifier head, not other layers *****")
        for name, param in model.named_parameters():
            if 'classifier' not in name: # only word embeddings
                param.requires_grad = False
        
    train_pipeline = HuggingFaceRoBERTaBase(tokenizer, 
                                            model, args.task_name, 
                                            TASK_CONFIG[args.task_name])
    logger.info(f"***** TASK NAME: {args.task_name} *****")
    
    # we use panda loader now, to make sure it is backward compatible
    # with our file writer.
    pd_format = False
    logger.info(f"***** Loading pre-loaded datasets from the disk directly! *****")
    datasets = DatasetDict.load_from_disk(args.train_file)

    def reverse_order(example):
        fields = TASK_CONFIG[args.task_name]
        for field in fields:
            if field:
                original_text = example[field]
                original_text = original_text.split(" ")[::-1]
                example[field] = " ".join(original_text)
        return example

    def random_order(example):
        fields = TASK_CONFIG[args.task_name]
        for field in fields:
            if field:
                original_text = example[field]
                original_text = original_text.split(" ")
                random.shuffle(original_text)
                example[field] = " ".join(original_text)
        return example

    if args.reverse_order:
        logger.warning("WARNING: you are reversing the order of your sequences.")
        datasets["train"] = datasets["train"].map(reverse_order)
        datasets["validation"] = datasets["validation"].map(reverse_order)
        datasets["test"] = datasets["test"].map(reverse_order)

    if args.random_order:
        logger.warning("WARNING: you are random ordering your sequences.")
        datasets["train"] = datasets["train"].map(random_order)
        datasets["validation"] = datasets["validation"].map(random_order)
        datasets["test"] = datasets["test"].map(random_order)
    # we don't care about test set in this script?

    if args.word_swapping:
        logger.warning("WARNING: performing word swapping.")
        # we need to do the swap on the data files.
        # this tokenizer helps you to get piece length for each token
        modified_tokenizer = ModifiedBertTokenizer(
            vocab_file="../data-files/bert_vocab.txt")
        modified_basic_tokenizer = ModifiedBasicTokenizer()
        datasets["train"] = datasets["train"].map(partial(random_corrupt, 
                                                       args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))
        datasets["validation"] = datasets["validation"].map(partial(random_corrupt, 
                                                       args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))
        datasets["test"] = datasets["test"].map(partial(random_corrupt, 
                                                       args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))

    # this may not always start for zero inoculation
    datasets["train"] = datasets["train"].shuffle(seed=args.seed)
    inoculation_train_df = datasets["train"]
    eval_df = datasets["validation"]
    # datasets["validation"] = datasets["validation"].shuffle(seed=args.seed)

    train_pipeline.train(inoculation_train_df, eval_df, 
                         model, args,
                         training_args, max_length=args.max_seq_length,
                         inoculation_patience_count=args.inoculation_patience_count, pd_format=pd_format, 
                         scramble_proportion=args.scramble_proportion, eval_with_scramble=args.eval_with_scramble)

