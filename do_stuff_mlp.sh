if [ ! -d 'onnx_files' ]; then   
    mkdir 'onnx_files'
fi

python run_evaluate_export.py --posenet_name mlp --keypoints gt --evaluate ./poseaug_baseline/poseaug/mlp/gt/poseaug/ckpt_best_dhp_p1.pth.tar  --output_onnx_name onnx_files/mlp_poseaug --device_selected cpu

if [ ! -d blob_files ]; then   
    mkdir blob_files
fi


cd blob_files
python ../blobconfiging.py -i ../onnx_files/gcn_poseaug.onnx -n 6 --cselector "wally"
