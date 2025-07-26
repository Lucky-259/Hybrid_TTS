#!/bin/bash 
# openr
num_sample_values=(4 8 16)
num_sequence_values=(4 8 16)
num_refine_values=(5 5 5)

prm_threshold_values=(0.0 0.0 0.0)
refine_cut_num_values=(0 0 0)
prm_gap_values=(0.0 0.0 0.0)

dataname="MATH500" #AIME24 MATH500 GPQA
modelname="Qwen2.5-3B-Instruct" #Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct gemma-3-4b-it
save_dir="results-solution/${modelname}/${dataname}"

LM=/path/.../Qwen2.5-3B-Instruct
RM=/path/.../Qwen2.5-Math-PRM-7B
task_name="critic_MATH"
rewrite_from="critic" #critic prm
max_new_tokens=2048
tree_max_width=4
tree_max_depth=4
method="critic_mcts"


for i in "${!num_sample_values[@]}"; do
    num_sample="${num_sample_values[$i]}"
    num_sequence="${num_sequence_values[$i]}"
    num_refine="${num_refine_values[$i]}"
    prm_threshold="${prm_threshold_values[$i]}"
    refine_cut_num="${refine_cut_num_values[$i]}"
    prm_gap="${prm_gap_values[$i]}"

    python -u reason/evaluation/eval.py \
    --LM $LM \
    --RM $RM \
    --task_name $task_name \
    --max_new_tokens $max_new_tokens \
    --num_sample $num_sample \
    --num_sequence $num_sequence \
    --num_refine $num_refine \
    --prm_threshold $prm_threshold \
    --refine_cut_num $refine_cut_num \
    --prm_gap $prm_gap \
    --rewrite_from $rewrite_from \
    --tree_max_width $tree_max_width \
    --tree_max_depth $tree_max_depth \
    --save_dir $save_dir \
    --method $method \
    > "/path/.../Hybrid_TTS/${save_dir}/${modelname}_${dataname}_${rewrite_from}_sample${num_sample}_seq${num_sequence}_refine${num_refine}_prm${prm_threshold}_@${refine_cut_num}gap${prm_gap}.log" 2>&1

    sleep 10
done