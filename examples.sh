# AdamSN heuristics
python torchrun_main.py --model_config configs/llama_60m.json --lr 5e-2 --batch_size 128 --total_batch_size 512 --num_training_steps 10000 --weight_decay 0 --dtype bfloat16 --eval_every 1000 --optimizer adamw_sn --use_subset_norm_heuristics  --scheduler cosine --warmup_steps 1000 --grad_clipping 1.0 --single_gpu
# AdamSN sqrt(d)/2
python torchrun_main.py --model_config configs/llama_60m.json --lr 5e-2 --batch_size 128 --total_batch_size 512 --num_training_steps 10000 --weight_decay 0 --dtype bfloat16 --eval_every 1000 --optimizer adamw_sn  --scheduler cosine --warmup_steps 1000 --grad_clipping 1.0 --single_gpu
# AdamSNSM heuristics
python torchrun_main.py --model_config=configs/llama_60m.json --batch_size=128 --total_batch_size=512 --use_subset_norm_heuristics --num_training_steps=10000 --weight_decay=0 --dtype=bfloat16 --eval_every=1000 --warmup_steps=1000 --scheduler=cosine --optimizer=adamw_snsm --lr=0.05 --update_proj_gap 200 --proj_type svd --rank=128 --update_proj_gap=100  --single_gpu --grad_clipping=1.0 --seed=420
# Adam
python torchrun_main.py --model_config configs/llama_60m.json --lr 5e-3 --batch_size 128 --total_batch_size 512 --num_training_steps 10000 --weight_decay 0 --dtype bfloat16 --eval_every 1000 --optimizer adamw --single_gpu --scheduler cosine --warmup_steps 1000
# RMSpropSN
python torchrun_main.py --model_config configs/llama_60m.json --beta1 0 --lr 1e-2 --batch_size 128 --total_batch_size 512 --num_training_steps 10000 --weight_decay 0 --dtype bfloat16 --eval_every 1000 --optimizer rmsprop_sn --use_subset_norm_heuristics  --scheduler cosine --warmup_steps 1000 --grad_clipping 1.0 --single_gpu
# AdaGradSNSM heuristics (default is with momentum 0.9)
python torchrun_main.py --model_config=configs/llama_60m.json --batch_size=128 --total_batch_size=512 --use_subset_norm_heuristics --num_training_steps=10000 --weight_decay=0 --dtype=bfloat16 --eval_every=1000 --warmup_steps=1000 --scheduler=cosine --optimizer=adagrad_snsm --lr=1 --update_proj_gap 200 --proj_type svd --rank=128 --update_proj_gap=100  --single_gpu --grad_clipping=1.0 --seed=420
# AdamSNSM with approximate svd
python torchrun_main.py --lr=0.01 --rank=128 --update_proj_gap=200 --model_config=configs/llama_60m.json --batch_size=128 --total_batch_size=512 --num_training_steps=10000 --weight_decay=0 --dtype=bfloat16 --eval_every=1000 --warmup_steps=1000 --scheduler=cosine --proj_type=svd --optimizer=adamw_snsm --single_gpu --grad_clipping=1.0 --seed=420 --approx_svd --asvd_srht_srank_scale 3
# AdamSNSM with SRHT (note: requires fast_hadamard_transform)
python torchrun_main.py --lr=0.05 --rank=128 --update_proj_gap=200 --model_config=configs/llama_60m.json --batch_size=128 --total_batch_size=512 --num_training_steps=10000 --weight_decay=0 --dtype=bfloat16 --eval_every=1000 --warmup_steps=1000 --scheduler=cosine --proj_type=srht --optimizer=adamw_snsm --single_gpu --grad_clipping=1.0 --seed=420