#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This script helps you to kick-off all experiments we
ran in a single script.
"""


# In[ ]:


GPU_TPYE = "REGULAR"
# LARGE means >30G GPU MEM


# In[ ]:


MID_TUNED_PATHS = [
"8-10_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0_reverse_False_random_True",
"8-4_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.01_reverse_True_random_False",
"8-4_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.05_reverse_True_random_False",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_0.01_reverse_False_random_False",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_0.05_reverse_False_random_False",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.01_reverse_False_random_True",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.05_reverse_False_random_True",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.2_reverse_False_random_True",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.2_reverse_True_random_False",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.4_reverse_True_random_False",
"8-5_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.8_reverse_True_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_0.8_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_0.01_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_0.05_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_0.2_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_0.4_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_0.8_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.01_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.05_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.4_reverse_False_random_False",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.4_reverse_False_random_True",
"8-6_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.8_reverse_False_random_True",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_1.0_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~ja_ktc@V_inoculation_1.0_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.05_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.2_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.4_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_0.8_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~ja_ktc@N~ja_ktc@V_inoculation_1.0_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.01_reverse_False_random_False",
"8-7_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.05_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_0.2_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M-en~fr@N~fr@V_inoculation_0.4_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.2_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.4_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_0.8_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0_reverse_False_random_False",
"8-9_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0_reverse_True_random_False", 
]


# In[ ]:


print("********************************************************")
print("*")
print("* Collecting Running Information")
print("*")
print("********************************************************")

YAML_FILE = input("Please type in your YAML file for this experiments (enter for command line setup): ")
if YAML_FILE.strip() != "":
    import yaml

    with open(YAML_FILE) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        experiment_cfg = yaml.load(file, Loader=yaml.FullLoader)
        GPU_ID = experiment_cfg["GPU_ID"]
        TASK_NAME = experiment_cfg["TASK_NAME"]
        NUM_TRAIN_EPOCHS = experiment_cfg["NUM_TRAIN_EPOCHS"]
        PER_DEVICE_BATCH_SIZE = experiment_cfg["PER_DEVICE_BATCH_SIZE"]
        PATIENT_COUNT = experiment_cfg["PATIENT_COUNT"]
        METRIC = experiment_cfg["METRIC"]
else:
    GPU_ID = input("Please type your GPU IDs separated by commas: ")
    TASK_NAME = input("Please type your fine-tuning task name: (sst3/snli/...): ")
    NUM_TRAIN_EPOCHS = input("Please give your NUM_TRAIN_EPOCHS: ")
    PER_DEVICE_BATCH_SIZE = input("Please give your PER_DEVICE_BATCH_SIZE: ")
    PATIENT_COUNT = input("Please give your PATIENT_COUNT: ")
    METRIC = input("Please give your metric for evaluating performance: (Macro-F1/accuracy/...): ")


# In[ ]:


print(f"TOTAL EXPERIMENTS: {len(MID_TUNED_PATHS)}")


# In[ ]:


import subprocess
experiments_run = [] 
for PATH in MID_TUNED_PATHS:
    print(f"EXPERIMENT {len(experiments_run)+1}/{len(MID_TUNED_PATHS)}")
    command = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run_fine_tuning.py                 --task_name {TASK_NAME}                 --model_name_or_path ../{PATH}/                 --metric_for_best_model {METRIC}                 --is_tensorboard                 --num_train_epochs {NUM_TRAIN_EPOCHS}                 --per_device_train_batch_size {PER_DEVICE_BATCH_SIZE}                 --inoculation_patience_count {PATIENT_COUNT}                 --eval_steps 50"
    print("EXPERIMENT COMMAND: ")
    print(command)
    list_dir = subprocess.Popen(command, shell=True)
    list_dir.wait()

