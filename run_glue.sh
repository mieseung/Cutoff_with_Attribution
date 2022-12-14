# Arguments
# $1: task name
# $2: number of GPU to use
# $3: batch size 
# $4: cutoff type

export GLUE_DIR=/home/jovyan/work/datasets
export TASK_NAME=$1
export NUM_GPU=$2
export BATCH_SIZE=$3
export CUTOFF_TYPE=$4
export EXCLUDE=$5
export CUTOFF_RATIO=$6

CUDA_VISIBLE_DEVICES=$NUM_GPU \
python run_glue.py \
  --model_name_or_path roberta-base \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_aug \
  --aug_type ${CUTOFF_TYPE}_cutoff \
  --aug_cutoff_ratio ${CUTOFF_RATIO} \
  --aug_ce_loss 1.0 \
  --aug_js_loss 1.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --logging_steps 500 \
  --save_steps 500 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --output_dir results/$TASK_NAME-$EXCLUDE-roberta_base-cutoff-$CUTOFF_RATIO \
  --overwrite_output_dir \
  --exclude_special_tokens