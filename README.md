# SSD: Single Shot MultiBox Detector in TensorFlow

SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). Present TF checkpoints have been directly converted from SSD Caffe models.



The [SSD Notebook](notebooks/ssd_notebook.ipynb) contains a minimal example of the SSD TensorFlow pipeline. Shortly, the detection is made of two main steps: running the SSD network on the image and post-processing the output using common algorithms (top-k filtering and Non-Maximum Suppression algorithm).

Here are two examples of successful detection outputs:
![](pictures/ex1.png "SSD anchors")
![](pictures/ex2.png "SSD anchors")

Here is a egocentric [video](http://www.dailymotion.com/video/x5if8ax), which is evaluated using SSD model

To run the notebook you first have to unzip the checkpoint files in ./checkpoint
```bash
unzip ssd_300_vgg.ckpt.zip
```
Install requirements
```bash
pip install -r requirements.txt
```
and then start a jupyter notebook with
```bash
jupyter notebook notebooks/ssd_notebook.ipynb
```
This is forked [respiratory](https://github.com/balancap/SSD-Tensorflow.git) and is fully fuctional version of SSD on Mac OS. 



