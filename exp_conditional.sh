python main.py --batch_size 32 --update_freq 1 --lr 0.0005 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset taxi --dataset_dir ./data/taxi --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.001 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset taobao --dataset_dir ./data/taobao --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.001 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset so --dataset_dir ./data/so --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.0001 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset retweet --dataset_dir ./data/retweet --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.005 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset mooc --dataset_dir ./data/mooc --transformer_dim 32 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0


python main.py --batch_size 32 --update_freq 1 --lr 0.0005 --epochs 50 --eval_every 2 --diffusion_steps 100 --sampling_steps 50 --gamma 0.99 --log_home . --dataset amazon --dataset_dir ./data/amazon --transformer_dim 16 --transformer_heads 4 --num_decoder_layers 4 --scheduler cosanneal --tgt_len 20 --block_size 4 --device cuda:0 --task conditional --seed 0