nohup python data_process.py --raw-root data/raw --output-root data/outputs/dataset > nohup-1.out 2>&1 &

nohup python depth_to_pointcloud.py --dataset-root data/outputs/dataset --overwrite > nohup-2.out 2>&1 &

# nohup  >  2>&1 &
