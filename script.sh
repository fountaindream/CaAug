The code is builded with DistributedDataParallel. 
Reprodecing the results in the paper should train the model on 2 GPUs.
You can also train this model on single GPU and double config.DATA.TRAIN_BATCH in configs.
For LTCC dataset
python -m torch.distributed.run --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/aug.yaml --gpu 0,1
# For PRCC dataset
python -m torch.distributed.run --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/aug.yaml --gpu 0,1 #
# # For VC-Clothes dataset. You should change the root path of '--resume' to your output path.
python -m torch.distributed.run --nproc_per_node=2 --master_port 8888 main.py --dataset vcclothes --cfg configs/aug.yaml --gpu 0,1 
# For DeepChange dataset. Using amp can accelerate training.
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset deepchange --cfg configs/aug.yaml --amp --gpu 0,1 #
# For LaST dataset. Using amp can accelerate training.
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset last --cfg configs/aug.yaml --amp --gpu 0,1 #
# For CCVID dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ccvid --cfg configs/aug.yaml --gpu 0,1 #
For market dataset
python -m torch.distributed.run --nproc_per_node=2 --master_port 8888 main.py --dataset market --cfg configs/aug.yaml --gpu 3,4