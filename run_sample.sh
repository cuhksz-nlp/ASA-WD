#train
python asa_wd_main.py --do_train --do_eval --train_file ./data/sample_data/train.txt --test_file ./data/sample_data/test.txt --bert_model ./bert-large-uncased/ --mem_valid aspect --dep_order second --num_epoch 10 --model_name ASA_WD.SAMPLE.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#test
python asa_wd_main.py --do_eval --model_path ./ASA_WD.SAMPLE.BERT.L/ --test_file ./data/sample_data/test.txt