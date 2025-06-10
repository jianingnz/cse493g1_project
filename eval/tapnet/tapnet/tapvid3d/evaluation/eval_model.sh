TAPVID3D_DIR="/home/jianing/research/cse493g1/eval/evaluation/ground_truth"
YOUR_PREDICTIONS_DIR="/home/jianing/research/cse493g1/eval/evaluation/predictions"

python3 evaluate_model.py \
  --tapvid3d_dir=$TAPVID3D_DIR \
  --tapvid3d_predictions=$YOUR_PREDICTIONS_DIR \
  --use_minival=True \
  --depth_scalings=median \
  --data_sources_to_evaluate=drivetrack \
  --debug=True