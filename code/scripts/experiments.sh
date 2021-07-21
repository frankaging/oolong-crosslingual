# notebook converter.
jupyter nbconvert --to python run_pos_tagging.ipynb
jupyter nbconvert --to python run_conllu.ipynb

# pos-tagging scripts.
CUDA_VISIBLE_DEVICES=6,7,8,9 python run_pos_tagging.py \
--batch_size 1000 \
--max_number_of_examples 1000

# conllu scripts.
CUDA_VISIBLE_DEVICES=6,7,8,9 python run_conllu.py \
--batch_size 500 \
--max_number_of_examples 1000

# galactic scripts.
GALACTIC_ROOT=./submodules/gdtreebank/ ./submodules/gdtreebank/bin/gd-translate \
--input ./data-files/wikitext-15M-conllu/wikitext-15M-train.conllu \
--spec en~fr@N~hi@V