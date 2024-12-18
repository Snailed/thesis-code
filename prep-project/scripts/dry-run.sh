#!/bin/bash

python main.py \
    --dry_run True \
    --dataset SineRegression \
    --experiment HMCInfer \
    --model SimpleBNN SimpleBNN2 SimpleBNN3 CirculantBNN CirculantBNN_medium_width CirculantBNN_large_width CirculantBNN_medium_width_3_deep CirculantBNN_medium_width_4_deep
#--model SimpleBNN SimpleBNN2 SimpleBNN3 CirculantBNN CirculantBNN_medium_width CirculantBNN_large_width CirculantBNN_medium_width_3_deep CirculantBNN_medium_width_4_deep