## SLFN

python main.py --model slfn --optimizer tusla --lr 0.5 --r .5 --beta 1e12 --total_epoch 200 --seed 111 --eta 1e-5
python main.py --model slfn --optimizer adam --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model slfn --optimizer amsgrad --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model slfn --optimizer rmsprop --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5

## TLFN

python main.py --model tlfn --optimizer tusla --lr 0.5 --r .5 --beta 1e12 --total_epoch 200 --seed 111 --eta 1e-5
python main.py --model tlfn --optimizer adam --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model tlfn --optimizer amsgrad --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model tlfn --optimizer rmsprop --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5