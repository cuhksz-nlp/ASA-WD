#train
python asa_wd_main.py --do_train --do_eval --train_file ./data/sample_data/train.txt --val_file ./data/sample_data/val.txt --test_file ./data/sample_data/test.txt --bert_model ./bert_base_uncased/ --mem_valid aspect --dep_order second --num_epoch 2 --model_path ./models/test_model --batch_size 2 --learning_rate 1e-5


#test
python asa_wd_main.py --do_eval --model_path ./ASA_WD.SAMPLE.BERT.L/ --test_file ./data/sample_data/test.txt
