#!/bin/bash

combos='global'

for combo in $combos
do 
  echo Running combo $combo
  python Train_Global_Model.py \
  --combo_id $combo \
  --experiment 'IMUPoserGlobalModel'  \
  --run_name "test" \
  --device "0" \
  --fast_dev_run
done
