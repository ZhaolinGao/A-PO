#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# running on 4x H100 GPUs
GPUS_PER_NODE=4
MICRO_BATCH_SIZE_PER_DEVICE=1 # reduce to reduce memory but slower
MODEL_SIZE=1.5 # 1.5B, 3B, 7B

# generation
MAX_PROMPT_LENGTH=256
MAX_RESPONSE_LENGTH=1024

# training config
FILTER_INCORRECT=False
batch_size=256
mini_batch_size=128
lr=1e-6
EPOCH=25
beta1=0.5
beta2=1e-3
num_gen_to_use=8

python3 -m verl.trainer.main_apo \
    algorithm.beta2=${beta2} \
    data.num_gen_to_use=${num_gen_to_use} \
    data.filter_incorrect=${FILTER_INCORRECT} \
    data.beta1=${beta1} \
    data.train_files=Cornell-AGI/math_size_qwen2.5_${MODEL_SIZE}b_eval \
    data.val_files=~/data/math/test.parquet \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.train_batch_size=${batch_size} \
    data.val_batch_size=500 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-${MODEL_SIZE}B \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZE_PER_DEVICE) \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZE_PER_DEVICE) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZE_PER_DEVICE) \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=apo \
    trainer.experiment_name=apo_math_${model_size} \
    trainer.total_epochs=${EPOCH}