
export CUDA_VISIBLE_DEVICES=1
bert_dir="bert-base-uncased"
task_name="rte"
exp_name=""

python run_glue.py\
  --do_train\
  --do_eval\
  --task_name $task_name\
  --model_name_or_path $bert_path\
  --output_dir out_dir/$task_name/$exp_name\
  --outputfile results/$task_name/$exp_name/results.csv\
  --model_type bert\
  --max_seq_length 128\
  --num_train_epochs 3\
  --overwrite_output_dir\
  --do_lower_case\
  --ib_dim 144\
  --deterministic\
  --learning_rate 2e-5\
  --eval_types dev train test\
  --evaluate_after_each_epoch\
  --seed 42\
