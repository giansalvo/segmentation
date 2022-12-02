#!/usr/bin/bash
python3 main.py train -m attnresunet -dr dataset_fake/ -w temp.h5 -c 3 -is 64 -b 2 -tl freeze_decoder
