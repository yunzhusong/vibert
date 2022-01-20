
export CUDA_VISIBLE_DEVICES=0

## RTE ===============================  
task_name="rte"
bert_dir="bert-base-uncased"
bert_dir="out_dir/mnli/vibert_1e-5_r2/checkpoint-6136_save"
#bert_dir="out_dir/$task_name/vibert_1e-5"

beta=1e-5
#exp_name="vibert_spur_prem_r2_$beta"
#exp_name="debug"
exp_name='trained_by_vibert_1e-5_r2'

python run_glue.py\
  --do_eval\
  --task_name $task_name\
  --model_name_or_path $bert_dir\
  --output_dir out_dir/$task_name/$exp_name\
  --outputfile out_dir/$task_name/$exp_name/results.csv\
  --model_type bert\
  --max_seq_length 128\
  --num_train_epochs 20\
  --overwrite_output_dir\
  --per_gpu_train_batch_size 32\
  --gradient_accumulation_steps 2\
  --do_lower_case\
  --ib_dim 384\
  --beta $beta\
  --ib\
  --learning_rate 2e-5\
  --eval_types dev\
  --kl_annealing linear\
  --evaluate_during_training\
  --evaluate_after_each_epoch\
  --logging_steps 100\
  --seed 812\
  --eval_tasks 'rte'\
  #--plotting_tsne\
  #--spurious_correlation premise_only\
  #--num_samples 6000\
  #--sample_train\
  #--do_train\

 
