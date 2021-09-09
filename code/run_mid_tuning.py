#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import torch
import random
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from datasets import DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# In[2]:


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    reinit_avg_embeddings: bool = field(
        default=False, metadata={"help": "Whether to reinit the embedding layer using the averaged embedding."},
    )
    reinit_embeddings: bool = field(
        default=False, metadata={"help": "Whether to reinit the embedding layer."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    inoculation_percentage: float = field(
        default=1.0, metadata={"help": "Ratio of training data to include in training."},
    )
    reverse_order: bool = field(
        default=False, metadata={"help": "Whether to reverse the sequence order."},
    )
    random_order: bool = field(
        default=False, metadata={"help": "Whether to random order the sequence."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            pass


# In[3]:


def main():
    
    os.environ["WANDB_PROJECT"] = "mid_tuning"
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
        
    logger.info("Generating the run name for WANDB for better experiment tracking.")
    import datetime
    date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)
    run_name = "{0}_{1}_{2}_seed_{3}_data_{4}_inoculation_{5}_reverse_{6}_random_{7}_reinit_emb_{8}_reinit_avg_{9}".format(
        date_time,
        model_args.model_name_or_path,
        model_args.tokenizer_name,
        training_args.seed,
        data_args.train_file.split("/")[-1].split(".")[0],
        data_args.inoculation_percentage,
        data_args.reverse_order,
        data_args.random_order,
        model_args.reinit_embeddings,
        model_args.reinit_avg_embeddings,
    )
    training_args.run_name = run_name
    logger.info(f"WANDB RUN NAME: {training_args.run_name}")
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    logger.info(f"OUTPUT DIR: {training_args.output_dir}")
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # We directly infer the steps to take upfront, so it is a fair 
    # comparision between different conditions of runs.
    
    # Our largest dataset is the original wiki-text data.
    # Loading it to memory to determine the steps.
    logger.info("Preloading largest dataset here to determine the step size.")
    LARGEST_DATASET = "../data-files/wikitext-15M/"
    wiki_datasets = DatasetDict.load_from_disk("../data-files/wikitext-15M/")
    NUM_MAX_STEP = (len(wiki_datasets["train"]))/(training_args.n_gpu*training_args.per_device_train_batch_size)
    # Overwrite.
    training_args.max_steps = int(NUM_MAX_STEP)
    logger.warning(f"SETTING: training_args.max_steps={training_args.max_steps}")
    
    training_args.warmup_steps = int(training_args.warmup_ratio*training_args.max_steps)
    
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
    logger.info("Training/evaluation parameters %s", training_args)

    logger.info("Training/evaluation parameters %s", data_args)
    
    logger.info("Training/evaluation parameters %s", model_args)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # this is our own tokenizer
    TOKENIZER_MAPPING = dict()
    TOKENIZER_MAPPING["bert"] = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=False,
        cache_dir=model_args.cache_dir
    )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    logger.info(f"***** Current setups *****")
    logger.info(f"***** model type: {model_args.model_name_or_path} *****")
    logger.info(f"***** tokenizer type: {model_args.tokenizer_name} *****")
    if model_args.tokenizer_name != model_args.model_name_or_path:
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
        random_model = AutoModelForMaskedLM.from_config(
            config=random_config,
        )
        random_model.resize_token_embeddings(len(tokenizer))
        replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data
        model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings

    if model_args.reinit_avg_embeddings:
        logger.info("***** WARNING: We reinit all embeddings to be the average embedding from the pretrained model. *****")
        replacing_embeddings = torch.mean(model.roberta.embeddings.word_embeddings.weight.data, dim=0).expand_as(model.roberta.embeddings.word_embeddings.weight.data)
        model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        # to keep consistent, we also need to reinit the type embeddings.
        random_model = AutoModelForMaskedLM.from_config(
            config=config,
        )
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    elif model_args.reinit_embeddings:
        logger.info("***** WARNING: We reinit all embeddings to be the randomly initialized embeddings. *****")
        random_model = AutoModelForMaskedLM.from_config(config)
        replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data
        model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        # to keep consistent, we also need to reinit the type embeddings.
        random_model = AutoModelForMaskedLM.from_config(
            config=config,
        )
        replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data
        model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    else:
        # do nothing.
        pass

    assert len(tokenizer) == model.roberta.embeddings.word_embeddings.weight.data.shape[0]
    
    # we also enhance this a little bit.
    # there are a couple of ways to
        
    # load from pre-processed wikitext data files
    if data_args.train_file is None:
        raise ValueError("This code requires a training/validation file.")

    datasets = DatasetDict.load_from_disk(data_args.train_file)
    
    if data_args.inoculation_percentage != 1.0:
        logger.info(f"WARNING: you are downsampling your training data with a ratio of {data_args.inoculation_percentage}.")
        inoculation_sample_size = int(len(datasets["train"]) * data_args.inoculation_percentage)
        datasets["train"] = datasets["train"].shuffle(seed=training_args.seed)
        inoculation_train_df = datasets["train"].select(range(inoculation_sample_size))
        # overwrite
        datasets["train"] = inoculation_train_df
    
    def reverse_order(example):
        original_text = example["text"]
        original_text = original_text.split(" ")[::-1]
        example["text"] = " ".join(original_text)
        return example
    
    def random_order(example):
        original_text = example["text"]
        original_text = original_text.split(" ")
        random.shuffle(original_text)
        example["text"] = " ".join(original_text)
        return example
    
    if data_args.reverse_order:
        logger.warning("WARNING: you are reversing the order of your sequences.")
        datasets["train"] = datasets["train"].map(reverse_order)
        datasets["validation"] = datasets["validation"].map(reverse_order)
        datasets["test"] = datasets["test"].map(reverse_order)

    if data_args.random_order:
        logger.warning("WARNING: you are random ordering your sequences.")
        datasets["train"] = datasets["train"].map(random_order)
        datasets["validation"] = datasets["validation"].map(random_order)
        datasets["test"] = datasets["test"].map(random_order)
    # we don't care about test set in this script?
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        print("****************************************")
        print("*                                      *")
        print("*  you are parsing line by line here!  *")
        print("*                                      *")
        print("****************************************")
        
        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        raise ValueError("I think parsing it line by line will be preferred. Please use --line_by_line.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # print out the model to see.
    print(model)
    
    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




