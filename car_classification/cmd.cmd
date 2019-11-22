python run_classifier.py --do_train=true --do_eval=true --dopredict=true --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=output --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=5 --train_batch_size=1 --learning_rate=5e-5 --num_train_epochs=3.0
python run_classifier.py --task_name=sim --do_predict=true --data_dir=data --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint = output --max_seq_length=5 --output_dir=result
python run_classifier.py --task_name=sim --do_predict=true --data_dir=data --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=output --max_seq_length=410 --output_dir=result

******************************
python run_classifier.py --do_train=true --do_eval=true --dopredict=true --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=output --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=512 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0

29/ 0.6938863 / content
python run_classifier.py --do_train=true --do_eval=true --dopredict=true --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=output --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=512 --train_batch_size=5 --learning_rate=2e-5 --num_train_epochs=3.0

29 / 0.6990167 / title
python run_classifier.py --do_train=true --do_eval=true --dopredict=false --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=output --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=410 --train_batch_size=5 --learning_rate=3e-5 --num_train_epochs=3.0


python run_classifier.py --do_train=true --do_eval=true --dopredict=true --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=output --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=512 --train_batch_size=32 --learning_rate=3e-5 --num_train_epochs=3.0
