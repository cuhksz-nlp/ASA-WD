#!/bin/bash

#LAPTOP
srun python asa_wd_main.py --do_eval --model_path ./release/ASA_WD.LAPTOP.BERT.L/ --test_file ./data/laptop/test.txt

#REST14
srun python asa_wd_main.py --do_eval --model_path ./release/ASA_WD.REST14.BERT.L/ --test_file ./data/rest14/test.txt

#REST15
srun python asa_wd_main.py --do_eval --model_path ./release/ASA_WD.REST15.BERT.L/ --test_file ./data/rest15/test.txt

#REST16
srun python asa_wd_main.py --do_eval --model_path ./release/ASA_WD.REST16.BERT.L/ --test_file ./data/rest16/test.txt

#TWITTER
srun python asa_wd_main.py --do_eval --model_path ./release/ASA_WD.TWITTER.BERT.L/ --test_file ./data/twitter/test.txt