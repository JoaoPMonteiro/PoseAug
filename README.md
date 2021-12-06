### PoseAug for the [OAK-D](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html)

Fork of the code [repository](https://github.com/jfzhang95/PoseAug/) for the paper:  
**PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation**  
[Kehong Gong](https://www.ece.nus.edu.sg/lv/index.html)\*, [Jianfeng Zhang](https://jeff95.me/)\*, [Jiashi Feng](https://sites.google.com/site/jshfeng/)  
CVPR 2021 (oral presentation)  
[[paper](https://arxiv.org/pdf/2105.02465.pdf)] 

with modifications to enable the pretrained models [provided](https://drive.google.com/drive/folders/1mLttbyZxsRdN5kw1IRdzZozyfndhV3Wh) by the aforementioned paper authors to be exported to onnx and blob formats

#### additional steps - required datasets (addapted from https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md)
   ```
   ${PoseAug}
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
   ${PoseAug}
   ├── data_extra
      ├── test_set
         ├── test_3dhp.npz
   ```

#### additional steps - environment 

check requirements.txt

#### generate onnx and blobs

> ./do_stuff.sh

#### output onnx and blobs

   ```
   ${PoseAug}
   ├── onnx_files
      ├── ...
   ├── blob_files
      ├── ...
   ```

### Acknowledgements

https://github.com/jfzhang95/

https://github.com/Garfield-kh

https://github.com/luxonis/

https://github.com/geaxgx/

https://github.com/PINTO0309/
