python main.py --batch_size 32 --update_freq 1 --lr 0.0005 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset taxi --dataset_dir ./data/taxi --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal  --block_size 8 --device cuda:0 --task unconditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.001 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset taobao --dataset_dir ./data/taobao --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal  --block_size 8 --device cuda:0 --task unconditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.01 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset so --dataset_dir ./data/so --transformer_dim 16 --transformer_heads 2 --num_decoder_layers 2 --scheduler cosanneal  --block_size 8 --device cuda:0 --task unconditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.0005 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset retweet --dataset_dir ./data/retweet --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal   --block_size 8 --device cuda:0 --task unconditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.005 --epochs 50 --eval_every 5 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset mooc --dataset_dir ./data/mooc --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal  --block_size 8 --device cuda:0 --task unconditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.01 --epochs 50 --eval_every 2 --diffusion_steps 50 --sampling_steps 50 --gamma 0.99 --log_home . --dataset amazon --dataset_dir ./data/amazon --transformer_dim 16 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal  --block_size 8 --device cuda:0 --task unconditional --seed 0