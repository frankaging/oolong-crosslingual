#!/usr/bin/env python
# coding: utf-8

# In[13]:


from glob import glob
import os


# In[ ]:


eval_method = "do_eval"
eval_model_path = "../stage_finetuned_models"
for path in glob(f"{eval_model_path}/*/"):
    print(f"generating results for path at: {path}")
    if "mnli" in path or "qqp" in path:
        cmd = f"CUDA_VISIBLE_DEVICES=9 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 16               --output_dir ../eval_finetuned_models"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES=9 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 32               --output_dir ../eval_finetuned_models"
    print(f"starting command")
    os.system(cmd)


# In[ ]:


# verification steps.


# In[28]:


# all_records = set([])
# path_record_map = {}
# for path in glob("../finetuned_models/*/"):
#     record = ("_".join(path.strip("/").split("/")[-1].split("_")[1:]))
#     all_records.add(record)
#     path_record_map[record] = path
    
# assert len(path_record_map) == len(all_records)

# import wandb
# api = wandb.Api()
# runs = api.runs("wuzhengx/big_transfer_eval")

# all_wandb_records = []
# for run in runs:
#     run_name = run.name
#     run_name = "_".join(run_name.split("_")[4:])
#     all_wandb_records.append(run_name)
    
# for r in all_records:
#     assert r in all_wandb_records:


# In[ ]:




