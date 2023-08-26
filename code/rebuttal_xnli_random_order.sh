for task in xnli-en xnli-es xnli-zh; do
        python "run_glue_xlmr.py" \
        --model_name_or_path "../99-99_xlm-roberta-base_xlm-roberta-base_seed_42_data_wikitext-15M_inoculation_0.0_reverse_False_random_True_reinit_emb_False_reinit_avg_False_token_s_False_word_s_False_reinit_cls_False_train_emb_False/" \
        --task_name "$task" \
        --do_train \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --output_dir ../ \
        --report_to none \
        --word_swapping True
done
