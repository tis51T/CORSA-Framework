@echo off
setlocal enabledelayedexpansion

for %%D in (13) do (
    for %%S in (7e-5) do (
        for %%L in (0.1) do (
            set "SEED=%%D"
            set "SLR=%%S"
            set "LOSS_LAMBDA=%%L"
            set "LOGDIR=/log/log_for_dual_prompts_multitasks_Aspect/15_%%D_aesc"
            set "DATAJSON=src/data/jsons/few_shot_for_prompt/twitter_2015/twitter15_%%D_info.json"
            set "CUDA_VISIBLE_DEVICES=0"
            echo !SLR!
            python MAESC_training_for_generated_dual_prompts_multitasks_Aspect.py ^
                --dataset twitter15 !DATAJSON! ^
                --checkpoint_dir ./ ^
                --model_config config/pretrain_base.json ^
                --log_dir !LOGDIR! ^
                --num_beams 4 ^
                --eval_every 1 ^
                --lr !SLR! ^
                --batch_size 4 ^
                --epochs 70 ^
                --grad_clip 5 ^
                --warmup 0.1 ^
                --num_workers 8 ^
                --has_prompt ^
                --num_image_tokens 4 ^
                --loss_lambda !LOSS_LAMBDA! ^
                --use_generated_aspect_prompt ^
                --use_different_aspect_prompt ^
                --use_multitasks ^
                --seed !SEED!
        )
    )
)
endlocal