# STD2P
Source code for "STD2P: RGBD Semantic Segmentation Using Spatio-Temporal Data-Driven Pooling (CVPR2017)"

Please visit the project page for all the information:
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/image-and-video-segmentation/rgbd-semantic-segmentation-using-spatio-temporal-data-driven-pooling/.

# Intallization & External softwares

   To run our model, you need to install our modified Caffe, and some external softwares.
   We use Epic Flow (https://thoth.inrialpes.fr/src/epicflow/) and RGBD version of MCG superpixel (https://github.com/s-gupta/rcnn-depth) in our method.
   In epic flow, we use RGBD version structured forest for edge detection (https://github.com/pdollar/edges).

# Test

1. Compute HHA representation for depth.

   We use HHA representation for depth, which transfer one channel representation into three channels.
   The source code of computing HHA representation can be found at https://github.com/s-gupta/rcnn-depth.

2. Compute superpixel and optical flow.

3. Compute region correspondence.

   We provide Matlab code to establish region correspondence in region_correspondence folder.
   Run the demo code run.m to see how it works.

4. Download the baseline model FCN-16s using RGB+HHA, and our pretrained model at http://datasets.d2.mpi-inf.mpg.de/yang17cvpr/STD2P_data.zip.

5. Generate the full model.
  
   python generate_full_network.py
   
   As a result, full_std2p_nyud2.caffemodel is generated in the folder examples/models.

6. Use predict.py to run our demo.

   python predict.py -g [gpu_id] -m [model]

# Training

   Due to GPU memory limitation, we cache pool4 and pool4_hha at the hard disk to provide more views.
   And then, we train the parameters after pool4. That is why we need to generate the full model at test stage 4.
   In our experiment, we provide 11 views to train the model.

   you can use examples/solve.py to train the model.

   python solve.py -g [gpu_id]

# Citation
If our work is useful for you, please consider citing:

@inproceedings{yang_cvpr17,

   title={STD2P: RGBD Semantic Segmentation Using Spatio-Temporal Data-Driven Pooling},
 
   author={Yang He and Wei-Chen Chiu and Margret Keuper and Mario Fritz},
 
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 
   year={2017}
 
}
