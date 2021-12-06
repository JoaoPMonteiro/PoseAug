if [ ! -d 'onnx_files' ]; then   
    mkdir 'onnx_files'
fi

if [ ! -f './onnx_files/videopose_poseaug.onnx' ]; then   
	python run_evaluate_export.py --posenet_name videopose --keypoints gt --evaluate ./poseaug_baseline/poseaug/videopose/gt/poseaug/ckpt_best_dhp_p1.pth.tar --output_onnx_name onnx_files/videopose_poseaug --device_selected cpu
fi
if [ ! -f './onnx_files/mlp_poseaug.onnx' ]; then   
	python run_evaluate_export.py --posenet_name mlp --keypoints gt --evaluate ./poseaug_baseline/poseaug/mlp/gt/poseaug/ckpt_best_dhp_p1.pth.tar  --output_onnx_name onnx_files/mlp_poseaug --device_selected cpu --stages 2
fi
if [ ! -f './onnx_files/stgcn_poseaug.onnx' ]; then   
	python run_evaluate_export.py --posenet_name stgcn --keypoints gt --evaluate ./poseaug_baseline/poseaug/stgcn/gt/poseaug/ckpt_best_dhp_p1.pth.tar --output_onnx_name onnx_files/stgcn_poseaug --device_selected cpu
fi
if [ ! -f './onnx_files/gcn_poseaug.onnx' ]; then   
	python run_evaluate_export.py --posenet_name gcn --keypoints gt --evaluate ./poseaug_baseline/poseaug/gcn/gt/poseaug/ckpt_best_dhp_p1.pth.tar  --output_onnx_name onnx_files/gcn_poseaug --device_selected cpu
fi
if [ ! -f './onnx_files/videopose_baseline.onnx' ]; then   
	python run_evaluate_export.py --posenet_name videopose --keypoints gt --evaluate ./poseaug_baseline/pretrain_baseline/videopose/gt/pretrain/ckpt_best.pth.tar --output_onnx_name onnx_files/videopose_baseline --device_selected cpu
fi
if [ ! -f './onnx_files/mlp_baseline.onnx' ]; then   
	python run_evaluate_export.py --posenet_name mlp --keypoints gt --evaluate ./poseaug_baseline/pretrain_baseline/mlp/gt/pretrain/ckpt_best.pth.tar  --output_onnx_name onnx_files/mlp_baseline --device_selected cpu --stages 2
fi
if [ ! -f './onnx_files/stgcn_baseline.onnx' ]; then   
	python run_evaluate_export.py --posenet_name stgcn --keypoints gt --evaluate ./poseaug_baseline/pretrain_baseline/stgcn/gt/pretrain/ckpt_best.pth.tar --output_onnx_name onnx_files/stgcn_baseline --device_selected cpu
fi
if [ ! -f './onnx_files/gcn_baseline.onnx' ]; then   
	python run_evaluate_export.py --posenet_name gcn --keypoints gt --evaluate ./poseaug_baseline/pretrain_baseline/gcn/gt/pretrain/ckpt_best.pth.tar --output_onnx_name onnx_files/gcn_baseline --device_selected cpu
fi






if [ ! -d blob_files ]; then   
    mkdir blob_files
fi

cd blob_files

python ../blobconfiging.py -i ../onnx_files/videopose_poseaug.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/mlp_poseaug.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/stgcn_poseaug.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/gcn_poseaug.onnx -n 6 --cselector "wally"

python ../blobconfiging.py -i ../onnx_files/videopose_baseline.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/mlp_baseline.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/stgcn_baseline.onnx -n 6 --cselector "wally"
python ../blobconfiging.py -i ../onnx_files/gcn_baseline.onnx -n 6 --cselector "wally"