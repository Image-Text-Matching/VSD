python train.py \
--batch_size 128 \
--data_path C:/datasets/data \
--dataset f30k \
--logger_name runs/f30k_test_1 \
--num_epochs 25 \
--mask_weight 1.0 \
--workers 0
python eval.py --dataset f30k --data_path C:/datasets/data
