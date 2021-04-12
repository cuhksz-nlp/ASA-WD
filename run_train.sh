#!/bin/bash


#LAPTOP
python asa_wd_main.py --do_train --train_file ./data/laptop/train.txt --test_file ./data/laptop/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.LAPTOP.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST14
python asa_wd_main.py --do_train --train_file ./data/rest14/train.txt --test_file ./data/rest14/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.REST14.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST15
python asa_wd_main.py --do_train --train_file ./data/rest15/train.txt --test_file ./data/rest15/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.REST15.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST16
python asa_wd_main.py --do_train --train_file ./data/rest16/train.txt --test_file ./data/rest16/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.REST16.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#TWITTER
python asa_wd_main.py --do_train --train_file ./data/twitter/train.txt --test_file ./data/twitter/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.TWITTER.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./
