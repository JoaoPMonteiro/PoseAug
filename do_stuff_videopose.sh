if [ ! -d 'onnx_files' ]; then   
    mkdir 'onnx_files'
fi

python run_evaluate_export.py --posenet_name videopose --keypoints gt --evaluate ./poseaug_baseline/poseaug/videopose/gt/poseaug/ckpt_best_dhp_p1.pth.tar --output_onnx_name onnx_files/videopose_poseaug --device_selected cpu
cd blob_files
python ../blobconfiging.py -i ../onnx_files/videopose_poseaug.onnx -n 6 --cselector "wally"
