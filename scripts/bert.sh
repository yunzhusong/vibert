export CUDA_VISIBLE_DEVICES=1
bert_dir="bert-base-uncased"

task_name="sts-b"
exp_name="bert_r3"
#bert_dir="out_dir/$task_name/bert"

python run_glue.py --do_train --do_eval\
  --task_name $task_name\
  --model_name_or_path $bert_dir\
  --num_train_epochs 25\
  --max_seq_length 128\
  --model_type bert\
  --overwrite_output_dir\
  --output_dir out_dir/$task_name/$exp_name\
  --outputfile results/$task_name/$exp_name/results.csv\
  --do_lower_case\
  --eval_types dev\
  --learning_rate 2e-5\
  --per_gpu_train_batch_size 16\
  --gradient_accumulation_steps 2\
  --evaluate_after_each_epoch\
  --seed 40\
  --logging_steps 50\
  --overwrite_output_dir\
  --evaluate_during_training\
  #--plotting_tsne\
  #--train_only classifier\
  #--spurious_correlation exchange\
  #--num_samples 200\
  #--sample_train\

