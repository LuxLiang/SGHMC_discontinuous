## optimizer = [SGHMC, SGLD, ADAM, RMSPROP, AMSGRAD]
python main.py --seed 111 --optimizer {optimizer} --lr 0.01 --dataset concrete --batch_size 128
python main.py --seed 222 --optimizer {optimizer} --lr 0.01 --dataset concrete --batch_size 128
python main.py --seed 333 --optimizer {optimizer} --lr 0.01 --dataset concrete --batch_size 128
python main.py --seed 444 --optimizer {optimizer} --lr 0.01 --dataset concrete --batch_size 128
python main.py --seed 555 --optimizer {optimizer} --lr 0.01 --dataset concrete --batch_size 128
