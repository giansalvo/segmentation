python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1b -w wt1b.h5 -lr 0.0001 -c 3
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1c -w wt1c.h5 -lr 0.0001 -c 3
python main.py train -m deeplabv3plus -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1d -w wt1d.h5 -lr 0.0001 -c 3
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1g -w wt1g.h5 -lr 0.0001 -c 3
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1i -w wt1i.h5 -lr 0.001 -c 3
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1m -w wt1m.h5 -lr 0.001 -c 3
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1o -w wt1o.h5 -lr 0.001 -c 4
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1o -w wt1o.h5 -lr 0.001 -c 4
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1p -w wt1p.h5 -lr 0.001 -c 3
python main.py train -m deeplabv3plus -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1s -w wt1s.h5 -lr 0.001 -c 3
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1u -w wt1u.h5 -lr 0.001 -c 3 -tl imagenet_freeze_down
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1v -w wt1v.h5 -lr 0.001 -c 3
python main.py train -m unet3 -dr .\dataset_pets\ -e 100 -b 64 -l logs_training1z -w wt1z.h5 -lr 0.001 -c 3

