#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
import os


# In[ ]:


eval_method = "do_eval"
for path in glob("../finetuned_models/*/"):
    print(f"generating results for path at: {path}")
    if "mnli" in path or "qqp" in path:
        cmd = f"CUDA_VISIBLE_DEVICES=9 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 16               --output_dir ../eval_finetuned_models"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES=9 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 32               --output_dir ../eval_finetuned_models"
    print(f"starting command")
    os.system(cmd)


# In[ ]:




