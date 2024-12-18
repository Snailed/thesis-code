#!/bin/bash

#--model SimpleBNN SimpleBNN2 SimpleBNN3 CirculantBNN CirculantBNN_medium_width CirculantBNN_large_width CirculantBNN_medium_width_3_deep CirculantBNN_medium_width_4_deep \
# python main.py \
#     --dataset SineRegression \
#     --experiment HMCInfer \
#     --num_samples=200 \
#     --model SimpleBNN SimpleBNN2 \
#     --num_warmup=100

python main.py \
    --dataset SineRegression \
    --experiment HMC \
    --num_samples=200 \
    --model SimpleBNN_1_deep CirculantBNN_medium_width_3_deep CirculantBNN_medium_width_4_deep \
    --num_warmup=500