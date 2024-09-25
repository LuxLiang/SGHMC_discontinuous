#!/bin/bash


# complete mkt
# num_asset = 5, m = 5
## hidden_size = 1, 5, 10
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sgd --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sgd --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sgd --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sgd --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sgd --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sgd --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step


#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sgld --lr 0.1 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sgld --lr 0.1 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sgld --lr 0.1 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sghmc --lr 0.01 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sghmc --lr 0.01 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sghmc --lr 0.01 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sghmc --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sghmc --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sghmc --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer sghmc --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer sghmc --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer sghmc --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer tusla --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer tusla --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer tusla --lr 0.5 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer tusla --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer tusla --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer tusla --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer adam --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer adam --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer adam --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step

#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 1 --optimizer amsgrad --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 5 --optimizer amsgrad --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
#python -u  main.py --asset_model BS --num_asset 5 --m 5 --strike_price 5 --num_path 20000 --u_gamma 0.5 --act_fn relu --hidden_size 10 --optimizer amsgrad --lr 0.05 --beta 1e12 --batch_size 128 --epochs 200 --scheduler --scheduler_type step
