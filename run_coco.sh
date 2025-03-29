python train.py \
--batch_size 256 \
--data_path C:/datasets/data \
--dataset coco \
--logger_name runs/coco_test_1 \
--num_epochs 25 \
--workers 0
python eval.py --dataset coco --data_path C:/datasets/data

