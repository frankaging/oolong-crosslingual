# structural shift
## control condition
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_mid_tuning.py \
--model_name_or_path roberta-base \
--tokenizer_name roberta-base \
--output_dir ../ \
--train_file ../data-files/wikitext-15M \
--do_train \
--do_eval \
--cache_dir ../.huggingface_cache/ \
--line_by_line \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 12 \
--logging_steps 25 \
--eval_steps 50 \
--seed 42 \
--save_total_limit 3 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--inoculation_percentage 1.0
# --random seeds = {42, 62, 82}
# --train_file = {
#    ../data-files/wikitext-15M,
#    ../data-files/wikitext-15M-en~fr@N~fr@V,
#    ../data-files/wikitext-15M-en~ja_ktc@N~ja_ktc@V,
#    ../data-files/wikitext-15M-en~fr@N~ja_ktc@V,
# }
# --reverse_order
# --random_order
# --inoculation_percentage ={0.2, 0.4, 0.6, 0.8, 1.0}
