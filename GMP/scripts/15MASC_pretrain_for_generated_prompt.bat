@echo off
setlocal enabledelayedexpansion

for %%D in (100) do (
    for %%S in (8e-5) do (
        for %%N in (4) do (
            set "SEED=%%D"
            set "SLR=%%S"
            set "NUM_IMAGE_TOKENS=%%N"
            set "LOGDIR=/log/log_for_generated_prompt/15_%%D_sc"
            set "DATAJSON=src/data/jsons/few_shot/twitter_2015/twitter15_%%D_info.json"
            set "CUDA_VISIBLE_DEVICES=0"
            python twitter_sc_training_for_generated_prompt.py ^
                --dataset twitter15 !DATAJSON! ^
                --checkpoint_dir ./ ^
                --model_config ./config/pretrain_base.json ^
                --log_dir !LOGDIR! ^
                --num_beams 4 ^
                --eval_every 1 ^
                --lr !SLR! ^
                --batch_size 4 ^
                --epochs 100 ^
                --grad_clip 5 ^
                --warmup 0.1 ^
                --is_sample 0 ^
                --seed !SEED! ^
                --num_image_tokens !NUM_IMAGE_TOKENS! ^
                --task twitter_sc ^
                --num_workers 8 ^
                --has_prompt ^
                --use_caption ^
                --use_generated_prompt ^
                --use_different_senti_prompt
        )
    )
)
endlocal