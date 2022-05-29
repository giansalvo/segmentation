REM unet
python main.py train -m unet -dr .\dataset_pets\ -e 100 -b 64 -lr 0.001 -l logs_training1p -w wt1p.h5
REM unet 2
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -lr 0.001 -l logs_training1u -w wt1u.h5 -tl imagenet_freeze_down
python main.py train -m unet2 -dr .\dataset_pets\ -e 100 -b 64 -lr 0.001 -l logs_training1v -w wt1v.h5
REM unet 3
python main.py train -m unet3 -dr .\dataset_pets\ -e 100 -b 64 -lr 0.001 -l logs_training1z -w wt1z.h5
