# cnn_polyp_detection
CNN framework for polyp detection in colonoscopy images. [Caffe](http://caffe.berkeleyvision.org/) framework is used to train three different Convolutional Neural Networks towards image segmentation for detecting polyps in colonoscopy frames.
The code was adapted from the Fully Convolutional Networks (FCN) of the following work

    Fully Convolutional Models for Semantic Segmentation
    Evan Shelhamer*, Jonathan Long*, Trevor Darrell
    PAMI 2016
    arXiv:1605.06211

    Fully Convolutional Models for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, Trevor Darrell
    CVPR 2015
    arXiv:1411.4038
    
and that is publicly available on this [github repo](https://github.com/shelhamer/fcn.berkeleyvision.org).

For this framework, I used three different CNNs, [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-), [GoogLeNet](https://arxiv.org/abs/1409.4842) and [VGG](https://arxiv.org/abs/1409.1556), in the FCN setting of [Long _et al._](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) and fine-tuned them on colonoscopy images towards image segmentation in polyp and non-polyp pixels. Polyp detection is achieved whenever the prediction region intersects the ground truth region.

Each CNN was trained on both RGB and RGBD (RGB+depth map) data. To extract the depth map of all data, the Shape-from-Shading technique of [Vinsentini-Scarzanella _et al._](http://www.commsp.ee.ic.ac.uk/~marcovs/shape-from-shading-for-metric-depth-reconstruction/) was used, with the C++ source code publicly available [here](http://www.commsp.ee.ic.ac.uk/~marcovs/shading/). In total there are 6 models. Each model can be found in the `models` folder and contains 9 basic files:
* **solve.py**: fine-tune the CNN using the training set
* **infer.py**: test the CNN using the test set
* **layers.py**: python data layer (RGB)
* **resume_solve.py**: continue solving from a previously saved state
* **score.py**: calculate the score during training
* **surgery.py**: perform basic operations like copying parameters and blob resizing
* **train.prototxt**: protobuf file with the CNN architecture for training
* **solver.prototxt**: protobuf file describing the Stochastic Gradient Descent solver for training the CNN
* **deploy.prototxt**: protobuf file with the CNN architecture for testing

Each RGBD model also includes a **nyud_layers.py** file that is an RGBD Python layer file and is used instead of **layers.py**.

## Training
To train a particular model, run the **solver.py** code in the relevant folder. Training invokes the **solver.prototxt** which initializes the model according to the **train.prototxt** file and applies backpropagation using the optimization method described in the solver file. The network weights are initialized by pre-trained weights. AlexNet and VGG are initialized by the `.caffemodel` files publicly available at [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org) and the GoogLeNet is initialized by the `.caffemodel` file publicly available at [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). These weight files are not uploaded into this repository. In order to train, please download the `.caffemodel` files and store them in the relevant **RGBD** model folders. The fine-tuned weights are stored in the `snapshot` folder.

## Testing
To test a particular model, run the **infer.py** code in the relevant folder. Testing initializes the architecture described in the **deploy.prototxt** file using the fine-tuned weights in the `snapshot` folder. Testing involves feeding all data in the test set and calculating the mean segmentation and detection precision and recall and the mean intersection over union.

## Datasets
Two databases were used as training set:
* [CVC-ClinicDB](http://www.medicalimagingandgraphics.com/article/S0895-6111(15)00056-7/abstract): publicly available [here](http://polyp.grand-challenge.org/site/Polyp/CVCClinicDB/)
* [ASU-Mayo Clinic Polyp Database](http://ieeexplore.ieee.org/document/7294676/): publicly available [here](http://polyp.grand-challenge.org/site/Polyp/AsuMayo/)  

and two databases were used as testing set:
* [ETIS-Larib Polyp DB](https://www.ncbi.nlm.nih.gov/pubmed/24037504): publicly available [here](http://polyp.grand-challenge.org/site/Polyp/EtisLarib/)
* [CVC-ColonDB](http://www.sciencedirect.com/science/article/pii/S0031320312001185): publicly available [here](http://mv.cvc.uab.es/projects/colon-qa/cvccolondb)

These datasets should be placed in the relevant `datasets` folder. After extracting the depth map from both the training and the test data, using the Shape-from-Shading technique, the depth maps should be placed inside the `datasets/depth` folder.
