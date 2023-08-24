first_arg="$1"
second_arg="$2"

for task in sst2 mrpc qnli cola mnli rte stsb wnli qqp; do
        python "run_glue_$first_arg.py" \
        --model_name_or_path "../99-99_${second_arg}_${second_arg}_seed_42_data_wikitext-15M_inoculation_0.0_reverse_False_random_False_reinit_emb_True_reinit_avg_False_token_s_False_word_s_False_reinit_cls_False_train_emb_True/" \
        --task_name "$task" \
        --do_train \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --output_dir ../ \
        --report_to none
done
