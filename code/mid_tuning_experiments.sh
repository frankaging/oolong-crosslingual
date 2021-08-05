# structural shift
## mid-tuning scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_mid_tuning.py \
--model_name_or_path roberta-base \
--tokenizer_name roberta-base \
--output_dir ../ \
--train_file ../data-files/wikitext-15M \
--do_train \
--cache_dir ../.huggingface_cache/ \
--line_by_line \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 12 \
--logging_steps 25 \
--seed 42 \
--save_total_limit 3 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--inoculation_percentage 1.0 \
--do_eval \
--eval_steps 50 \
--evaluation_strategy steps
# --random seeds = {42, 62, 82}
# --train_file = {
#    ../data-files/wikitext-15M,
#    ../data-files/wikitext-15M-en~fr@N~fr@V,
#    ../data-files/wikitext-15M-en~ja_ktc@N~ja_ktc@V,
#    ../data-files/wikitext-15M-en~fr@N~ja_ktc@V,
# }
# --reverse_order
# --random_order
# --inoculation_percentage ={0.01, 0.05, 0.2, 0.4, 0.8, 1.0}

## fine-tuning scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_fine_tuning.py \
--train_file ../data-files/sst3 \
--model_name_or_path ../7-31_roberta-base_roberta-base_seed_42_data_wikitext-15M_inoculation_1.0_reverse_False_random_False/ \
--output_dir ../ \
--metric_for_best_model Macro-F1 \
--is_tensorboard \
--num_train_epochs 10
# --train_file = {
#    # there are the regular files
#    ../data-files/sst3,
#    ../data-files/snli,
#    ../data-files/qnli,
#    ../data-files/mrpc,
#    ../data-files/cola,
#    # we also have shifted files
# }
# --model_name_or_path [YOUR_SAVED_MODEL_ONLY]

