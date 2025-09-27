@echo off
setlocal

set loss_lambda=0.1
set sl=6.5e-5

for %%S in (100) do (
    set "seed=%%S"
    set "num_image_tokens=4"
    set "CUDA_VISIBLE_DEVICES=0"
    call python twitter_ae_training_for_generated_prompt_multitasks.py ^
        --dataset twitter15 ./src/data/jsons/few_shot_for_prompt/twitter_2015/twitter15_%%S_info.json ^
        --checkpoint_dir ./ ^
        --model_config ./config/pretrain_base.json ^
        --log_dir /log/log_for_generated_aspect_prompt_multitasks/15_%%S_ae ^
        --num_beams 4 ^
        --eval_every 1 ^
        --lr %sl% ^
        --batch_size 4 ^
        --epochs 50 ^
        --grad_clip 5 ^
        --warmup 0.1 ^
        --is_sample 0 ^
        --seed %%S ^
        --task twitter_ae ^
        --num_workers 8 ^
        --num_image_tokens 4 ^
        --loss_lambda %loss_lambda% ^
        --use_multitasks ^
        --has_prompt ^
        --use_generated_prompt ^
        --use_different_aspect_prompt
)

endlocal