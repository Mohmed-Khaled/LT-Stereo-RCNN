# Stereo R-CNN
## Stereo R-CNN based 3D Object Detection for Autonomous Driving

<img src="doc/system.png" width = 90% div align=center />

Stereo R-CNN focuses on accurate 3D object detection and estimation using image-only data in autonomous driving scenarios. It features simultaneous object detection and association for stereo images, 3D box estimation using 2D information, accurate dense alignment for 3D box refinement.

## 1.0 Install

This implementation is tested with Anaconda Python 3.6 and Pytorch 1.7.0

1.1 Create Conda Environment:
```
    conda create -n env_stereo python=3.6
    conda activate env_stereo
```
1.2 Install PyTorch 1.7.0

Go to https://pytorch.org/ and choose your cuda version and run the command generated from the website

1.3 Clone Repo, Install Requirements, and Build:
```
    git clone https://github.com/Mohmed-Khaled/LT-Stereo-RCNN.git
    cd LT-Stereo-RCNN
    pip install -r requirements.txt
    ./build.sh
```

## 2.0 Quick Demo

2.1 Create the folder for placing the model
```
    mkdir models_stereo
```
2.2 Download pretrained model [One Drive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/pliap_connect_ust_hk/ETowGlAcACJBman4TnBO5LMB2-9NQPf1h_8Cd9Xl0HUIkQ?e=QmXwJC)/[Google Drive](https://drive.google.com/file/d/1rIS43NzTvjRMX9m3UZIG5EvgFzXOVZWX/view?usp=sharing) and put it into models_stereo/

2.3 Run Demo

```
    python demo.py
```

If everything goes well, you will see the detection result on the left, right and bird's eye view image respectively.

Some sample results:

<img src="doc/results.png" width = 90% div align=center />

## 3.0 Dataset Preparation

3.1 Download the left image, right image, calibration, labels and point clouds (optional for visualization) from [KITTI Object Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Make sure the structure looks like:
```
yourfolder/
        training
        image_2
        image_3
        label_2
        calib
        velodyne
```
3.2 Create symlinks:
```
    cd data/kitti
    ln -s yourfolder object
    cd ../..
```
## 4.0 Training

Download the Res-101 pretrained weight [One Drive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/pliap_connect_ust_hk/ESxYZL8TjzxAneYh_mWflFsBs0lZ-hc4m67W1k9_hYgXzQ?e=dZdoej)/[Google Drive](https://drive.google.com/file/d/1_t8TtUevtMdnvZ2SoD7Ut8sS1adyCKTt/view?usp=sharing), and put it into data/pretrained_model

Set corresponding CUDA_VISIBLE_DEVICES in train.sh, and simply run
```
    ./train.sh
```
It consumes ~11G GPU memery for training.

The trained model and training log are saved in /models_stereo by default.

## 5.0 Evaluation

You can evaluate the 3D detection performance using either our provided model or your trained model. 
Set corresponding CUDA_VISIBLE_DEVICES in test.sh, and run
```
    ./test.sh
```

The results are saved in models_stereo/result/data by default.

To evalute average precision for your result run this script
```
    ./eval_results.sh
```
The evaluation code is based on the object development kit provided by [KITTI Object Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## 6.0 Acknowledgments

This project is based on [Stereo R-CNN](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN) by [Peiliang Li](https://peiliangli.github.io/), [Xiaozhi Chen](https://xiaozhichen.github.io/) and [Shaojie Shen](http://uav.ust.hk/group/) from the [HKUST Aerial Robotics Group](http://uav.ust.hk/), and [DJI](https://www.dji.com/).

## 7.0 License
The source code is released under [MIT](https://opensource.org/licenses/MIT) license.
