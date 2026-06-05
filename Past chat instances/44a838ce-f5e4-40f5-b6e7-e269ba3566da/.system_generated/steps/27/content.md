Title: Live Content

Description: Fetched live

Source: https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md

---

Awesome-Pytorch-list
========================

![pytorch-logo-dark](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

<p align="center">
	<img src="https://img.shields.io/badge/stars-12400+-brightgreen.svg?style=flat"/>
	<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
</p>

## Contents
- [Pytorch & related libraries](#pytorch--related-libraries)
  - [NLP & Speech Processing](#nlp--Speech-Processing)
  - [Computer Vision](#cv)
  - [Probabilistic/Generative Libraries](#probabilisticgenerative-libraries)
  - [Other libraries](#other-libraries)
- [Tutorials, books & examples](#tutorials-books--examples)
- [Paper implementations](#paper-implementations)
- [Talks & Conferences](#talks--conferences)
- [Pytorch elsewhere](#pytorch-elsewhere)

## Pytorch & related libraries

1. [pytorch](http://pytorch.org): Tensors and Dynamic neural networks in Python with strong GPU acceleration.
2. [Captum](https://github.com/pytorch/captum): Model interpretability and understanding for PyTorch.

### NLP & Speech Processing:

1. [pytorch text](https://github.com/pytorch/text): Torch text related contents.  
2. [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq): A framework for sequence-to-sequence (seq2seq) models implemented in PyTorch.  
3. [anuvada](https://github.com/Sandeep42/anuvada): Interpretable Models for NLP using PyTorch.
4. [audio](https://github.com/pytorch/audio): simple audio I/O for pytorch.
5. [loop](https://github.com/facebookresearch/loop): A method to generate speech across multiple speakers
6. [fairseq-py](https://github.com/facebookresearch/fairseq-py): Facebook AI Research Sequence-to-Sequence Toolkit written in Python.
7. [speech](https://github.com/awni/speech): PyTorch ASR Implementation.
8. [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py): Open-Source Neural Machine Translation in PyTorch http://opennmt.net 
9. [neuralcoref](https://github.com/huggingface/neuralcoref): State-of-the-art coreference resolution based on neural nets and spaCy huggingface.co/coref
10. [sentiment-discovery](https://github.com/NVIDIA/sentiment-discovery): Unsupervised Language Modeling at scale for robust sentiment classification.
11. [MUSE](https://github.com/facebookresearch/MUSE): A library for Multilingual Unsupervised or Supervised word Embeddings
12. [nmtpytorch](https://github.com/lium-lst/nmtpytorch): Neural Machine Translation Framework in PyTorch.
13. [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet): An implementation of WaveNet with fast generation
14. [Tacotron-pytorch](https://github.com/soobinseo/Tacotron-pytorch): Tacotron: Towards End-to-End Speech Synthesis.
15. [AllenNLP](https://github.com/allenai/allennlp): An open-source NLP research library, built on PyTorch.
16. [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP): Text utilities and datasets for PyTorch pytorchnlp.readthedocs.io
17. [quick-nlp](https://github.com/outcastofmusic/quick-nlp): Pytorch NLP library based on FastAI. 
18. [TTS](https://github.com/mozilla/TTS): Deep learning for Text2Speech
19. [LASER](https://github.com/facebookresearch/LASER): Language-Agnostic SEntence Representations
20. [pyannote-audio](https://github.com/pyannote/pyannote-audio): Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding
21. [gensen](https://github.com/Maluuba/gensen): Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning.
22. [translate](https://github.com/pytorch/translate): Translate - a PyTorch Language Library.
23. [espnet](https://github.com/espnet/espnet): End-to-End Speech Processing Toolkit espnet.github.io/espnet
24. [pythia](https://github.com/facebookresearch/pythia): A software suite for Visual Question Answering
25. [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT): Phrase-Based & Neural Unsupervised Machine Translation.
26. [jiant](https://github.com/jsalt18-sentence-repl/jiant): The jiant sentence representation learning toolkit. 
27. [BERT-PyTorch](https://github.com/codertimo/BERT-pytorch): Pytorch implementation of Google AI's 2018 BERT, with simple annotation
28. [InferSent](https://github.com/facebookresearch/InferSent): Sentence embeddings (InferSent) and training code for NLI.
29. [uis-rnn](https://github.com/google/uis-rnn):This is the library for the Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm, corresponding to the paper Fully Supervised Speaker Diarization. arxiv.org/abs/1810.04719 
30. [flair](https://github.com/zalandoresearch/flair): A very simple framework for state-of-the-art Natural Language Processing (NLP)
31. [pytext](https://github.com/facebookresearch/pytext): A natural language modeling framework based on PyTorch fb.me/pytextdocs
32. [voicefilter](https://github.com/mindslab-ai/voicefilter): Unofficial PyTorch implementation of Google AI's VoiceFilter system http://swpark.me/voicefilter
33. [BERT-NER](https://github.com/kamalkraj/BERT-NER): Pytorch-Named-Entity-Recognition-with-BERT. 
34. [transfer-nlp](https://github.com/feedly/transfer-nlp): NLP library designed for flexible research and development
35. [texar-pytorch](https://github.com/asyml/texar-pytorch): Toolkit for Machine Learning and Text Generation, in PyTorch texar.io
36. [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi): pytorch-kaldi is a project for developing state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN part is managed by pytorch, while feature extraction, label computation, and decoding are performed with the kaldi toolkit.
37. [NeMo](https://github.com/NVIDIA/NeMo): Neural Modules: a toolkit for conversational AI nvidia.github.io/NeMo
38. [pytorch-struct](https://github.com/harvardnlp/pytorch-struct): A library of vectorized implementations of core structured prediction algorithms (HMM, Dep Trees, CKY, ..,)
39. [espresso](https://github.com/freewym/espresso): Espresso: A Fast End-to-End Neural Speech Recognition Toolkit
40. [transformers](https://github.com/huggingface/transformers): huggingface Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch. huggingface.co/transformers
41. [reformer-pytorch](https://github.com/lucidrains/reformer-pytorch): Reformer, the efficient Transformer, in Pytorch
42. [torch-metrics](https://github.com/enochkan/torch-metrics): Metrics for model evaluation in pytorch
43. [speechbrain](https://github.com/speechbrain/speechbrain): SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch.
44. [Backprop](https://github.com/backprop-ai/backprop): Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.

### CV:

1. [pytorch vision](https://github.com/pytorch/vision): Datasets, Transforms and Models specific to Computer Vision.
2. [pt-styletransfer](https://github.com/tymokvo/pt-styletransfer): Neural style transfer as a class in PyTorch.
3. [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch):  PyTorch module to use OpenFace's nn4.small2.v1.t7 model
4. [img_classification_pk_pytorch](https://github.com/felixgwu/img_classification_pk_pytorch): Quickly comparing your image classification models with the state-of-the-art models (such as DenseNet, ResNet, ...)
5. [SparseConvNet](https://github.com/facebookresearch/SparseConvNet): Submanifold sparse convolutional networks.
6. [Convolution_LSTM_pytorch](https://github.com/automan000/Convolution_LSTM_pytorch): A multi-layer convolution LSTM module
7. [face-alignment](https://github.com/1adrianb/face-alignment): :fire: 2D and 3D Face alignment library build using pytorch adrianbulat.com
8. [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation): PyTorch for Semantic Segmentation.
9. [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch): This is a PyTorch version of RoIAlign. This implementation is based on crop_and_resize and supports both forward and backward on CPU and GPU.
10. [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune): Fine-tune pretrained Convolutional Neural Networks with PyTorch.
11. [detectorch](https://github.com/ignacio-rocco/detectorch): Detectorch - detectron for PyTorch
12. [Augmentor](https://github.com/mdbloice/Augmentor): Image augmentation library in Python for machine learning. http://augmentor.readthedocs.io
13. [s2cnn](https://github.com/jonas-koehler/s2cnn): 
This library contains a PyTorch implementation of the SO(3) equivariant CNNs for spherical signals (e.g. omnidirectional cameras, signals on the globe)
14. [TorchCV](https://github.com/donnyyou/torchcv): A PyTorch-Based Framework for Deep Learning in Computer Vision. 
15. [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark): Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.
16. [image-classification-mobile](https://github.com/osmr/imgclsmob): Collection of classification models pretrained on the ImageNet-1K.
17. [medicaltorch](https://github.com/perone/medicaltorch): A medical imaging framework for Pytorch http://medicaltorch.readthedocs.io
18. [albumentations](https://github.com/albu/albumentations): Fast image augmentation library.
19. [kornia](https://github.com/arraiyopensource/kornia): Differentiable computer vision library.
20. [pytorch-text-recognition](https://github.com/s3nh/pytorch-text-recognition): Text recognition combo - CRAFT + CRNN.
21. [facenet-pytorch](https://github.com/timesler/facenet-pytorch): Pretrained Pytorch face detection and recognition models ported from davidsandberg/facenet.
22. [detectron2](https://github.com/facebookresearch/detectron2): Detectron2 is FAIR's next-generation research platform for object detection and segmentation.
23. [vedaseg](https://github.com/Media-Smart/vedaseg): A semantic segmentation framework by pyotrch
24. [ClassyVision](https://github.com/facebookresearch/ClassyVision): An end-to-end PyTorch framework for image and video classification.
25. [detecto](https://github.com/alankbi/detecto):Computer vision in Python with less than 10 lines of code
26. [pytorch3d](https://github.com/facebookresearch/pytorch3d): PyTorch3D is FAIR's library of reusable components for deep learning with 3D data pytorch3d.org
27. [MMDetection](https://github.com/open-mmlab/mmdetection): MMDetection is an open source object detection toolbox, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
28. [neural-dream](https://github.com/ProGamerGov/neural-dream): A PyTorch implementation of the DeepDream algorithm. Creates dream-like hallucinogenic visuals.
29. [FlashTorch](https://github.com/MisaOgura/flashtorch): Visualization toolkit for neural networks in PyTorch!
30. [Lucent](https://github.com/greentfrapp/lucent): Tensorflow and OpenAI Clarity's Lucid adapted for PyTorch.
31. [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): MMDetection3D is OpenMMLab's next-generation platform for general 3D object detection, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
32. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): MMSegmentation is a semantic segmentation toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
33. [MMEditing](https://github.com/open-mmlab/mmediting): MMEditing is a image and video editing toolbox, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
34. [MMAction2](https://github.com/open-mmlab/mmaction2): MMAction2 is OpenMMLab's next generation action understanding toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
35. [MMPose](https://github.com/open-mmlab/mmpose): MMPose is a pose estimation toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
36. [lightly](https://github.com/lightly-ai/lightly) - Lightly is a computer vision framework for self-supervised learning.
37. [RoMa](https://naver.github.io/roma/): a lightweight and efficient library to deal with 3D rotations.


### Probabilistic/Generative Libraries:

1. [ptstat](https://github.com/stepelu/ptstat): Probabilistic Programming and Statistical Inference in PyTorch
2. [pyro](https://github.com/uber/pyro): Deep universal probabilistic programming with Python and PyTorch http://pyro.ai
3. [probtorch](https://github.com/probtorch/probtorch): Probabilistic Torch is library for deep generative models that extends PyTorch.
4. [paysage](https://github.com/drckf/paysage): Unsupervised learning and generative models in python/pytorch.
5. [pyvarinf](https://github.com/ctallec/pyvarinf): Python package facilitating the use of Bayesian Deep Learning methods with Variational Inference for PyTorch. 
6. [pyprob](https://github.com/probprog/pyprob): A PyTorch-based library for probabilistic programming and inference compilation.
7. [mia](https://github.com/spring-epfl/mia): A library for running membership inference attacks against ML models. 
8. [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): ProGAN package implemented as an extension of PyTorch nn.Module.
9. [botorch](https://github.com/pytorch/botorch): Bayesian optimization in PyTorch

### Other libraries:

1. [pytorch extras](https://github.com/mrdrozdov/pytorch-extras): Some extra features for pytorch.    
2. [functional zoo](https://github.com/szagoruyko/functional-zoo): PyTorch, unlike lua torch, has autograd in it's core, so using modular structure of torch.nn modules is not necessary, one can easily allocate needed Variables and write a function that utilizes them, which is sometimes more convenient. This repo contains model definitions in this functional way, with pretrained weights for some models. 
3. [torch-sampling](https://github.com/ncullen93/torchsample): This package provides a set of transforms and data structures for sampling from in-memory or out-of-memory data. 
4. [torchcraft-py](https://github.com/deepcraft/torchcraft-py): Python wrapper for TorchCraft, a bridge between Torch and StarCraft for AI research.
5. [aorun](https://github.com/ramon-oliveira/aorun): Aorun intend to be a Keras with PyTorch as backend. 
6. [logger](https://github.com/oval-group/logger): A simple logger for experiments.
7. [PyTorch-docset](https://github.com/iamaziz/PyTorch-docset): PyTorch docset! use with Dash, Zeal, Velocity, or LovelyDocs.  
8. [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch): Convert torch t7 model to pytorch model and source.
9. [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch): The goal of this repo is to help to reproduce research papers results.  
10. [pytorch_fft](https://github.com/locuslab/pytorch_fft): PyTorch wrapper for FFTs
11. [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
12. [pytorch-extension](https://github.com/sniklaus/pytorch-extension): This is a CUDA extension for PyTorch which computes the Hadamard product of two tensors.
13. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): This module saves PyTorch tensors in tensorboard format for inspection. Currently supports scalar, image, audio, histogram features in tensorboard.
14. [gpytorch](https://github.com/jrg365/gpytorch): GPyTorch is a Gaussian Process library, implemented using PyTorch. It is designed for creating flexible and modular Gaussian Process models with ease, so that you don't have to be an expert to use GPs.
15. [spotlight](https://github.com/maciejkula/spotlight): Deep recommender models using PyTorch.
16. [pytorch-cns](https://github.com/awentzonline/pytorch-cns): Compressed Network Search with PyTorch
17. [pyinn](https://github.com/szagoruyko/pyinn): CuPy fused PyTorch neural networks ops
18. [inferno](https://github.com/nasimrahaman/inferno): A utility library around PyTorch
19. [pytorch-fitmodule](https://github.com/henryre/pytorch-fitmodule): Super simple fit method for PyTorch modules
20. [inferno-sklearn](https://github.com/dnouri/inferno): A scikit-learn compatible neural network library that wraps pytorch.
21. [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert): convert between pytorch, caffe prototxt/weights and darknet cfg/weights
22. [pytorch2caffe](https://github.com/longcw/pytorch2caffe): Convert PyTorch model to Caffemodel
23. [pytorch-tools](https://github.com/nearai/pytorch-tools): Tools for PyTorch
24. [sru](https://github.com/taolei87/sru): Training RNNs as Fast as CNNs (arxiv.org/abs/1709.02755)
25. [torch2coreml](https://github.com/prisma-ai/torch2coreml): Torch7 -> CoreML
26. [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding): PyTorch Deep Texture Encoding Network http://hangzh.com/PyTorch-Encoding
27. [pytorch-ctc](https://github.com/ryanleary/pytorch-ctc): PyTorch-CTC is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch. C++ code borrowed liberally from TensorFlow with some improvements to increase flexibility.
28. [candlegp](https://github.com/t-vi/candlegp): Gaussian Processes in Pytorch. 
29. [dpwa](https://github.com/loudinthecloud/dpwa): Distributed Learning by Pair-Wise Averaging. 
30. [dni-pytorch](https://github.com/koz4k/dni-pytorch): Decoupled Neural Interfaces using Synthetic Gradients for PyTorch.
31. [skorch](https://github.com/dnouri/skorch): A scikit-learn compatible neural network library that wraps pytorch
32. [ignite](https://github.com/pytorch/ignite): Ignite is a high-level library to help with training neural networks in PyTorch.
33. [Arnold](https://github.com/glample/Arnold): Arnold - DOOM Agent
34. [pytorch-mcn](https://github.com/albanie/pytorch-mcn): Convert models from MatConvNet to PyTorch
35. [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch): A simplified implemention of Faster R-CNN with competitive performance.
36. [generative_zoo](https://github.com/DL-IT/generative_zoo): generative_zoo is a repository that provides working implementations of some generative models in PyTorch.
37. [pytorchviz](https://github.com/szagoruyko/pytorchviz): A small package to create visualizations of PyTorch execution graphs. 
38. [cogitare](https://github.com/cogitare-ai/cogitare): Cogitare - A Modern, Fast, and Modular Deep Learning and Machine Learning framework in Python. 
39. [pydlt](https://github.com/dmarnerides/pydlt): PyTorch based Deep Learning Toolbox
40. [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch): Implementations of different VAE-based semi-supervised and generative models in PyTorch. 
41. [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster): PyTorch Extension Library of Optimised Graph Cluster Algorithms.
42. [neural-assembly-compiler](https://github.com/aditya-khant/neural-assembly-compiler): A neural assembly compiler for pyTorch based on adaptive-neural-compilation. 
43. [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch): Convert Caffe models to PyTorch.
44. [extension-cpp](https://github.com/pytorch/extension-cpp): C++ extensions in PyTorch
45. [pytoune](https://github.com/GRAAL-Research/pytoune): A Keras-like framework and utilities for PyTorch
46. [jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement): Deep reinforcement learning libraries for NVIDIA Jetson TX1/TX2 with PyTorch, OpenAI Gym, and Gazebo robotics simulator.
47. [matchbox](https://github.com/salesforce/matchbox): Write PyTorch code at the level of individual examples, then run it efficiently on minibatches.
48. [torch-two-sample](https://github.com/josipd/torch-two-sample): A PyTorch library for two-sample tests
49. [pytorch-summary](https://github.com/sksq96/pytorch-summary): Model summary in PyTorch similar to `model.summary()` in Keras
50. [mpl.pytorch](https://github.com/BelBES/mpl.pytorch): Pytorch implementation of MaxPoolingLoss.
51. [scVI-dev](https://github.com/YosefLab/scVI-dev): Development branch of the scVI project in PyTorch
52. [apex](https://github.com/NVIDIA/apex): An Experimental PyTorch Extension(will be deprecated at a later point)
53. [ELF](https://github.com/pytorch/ELF): ELF: a platform for game research.
54. [Torchlite](https://github.com/EKami/Torchlite): A high level library on top of(not only) Pytorch
55. [joint-vae](https://github.com/Schlumberger/joint-vae): Pytorch implementation of JointVAE, a framework for disentangling continuous and discrete factors of variation star2
56. [SLM-Lab](https://github.com/kengz/SLM-Lab): Modular Deep Reinforcement Learning framework in PyTorch.
57. [bindsnet](https://github.com/Hananel-Hazan/bindsnet): A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using PyTorch
58. [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): ProGAN package implemented as an extension of PyTorch nn.Module
59. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): Geometric Deep Learning Extension Library for PyTorch
60. [torchplus](https://github.com/knighton/torchplus): Implements the + operator on PyTorch modules, returning sequences.
61. [lagom](https://github.com/zuoxingdong/lagom): lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.
62. [torchbearer](https://github.com/ecs-vlc/torchbearer): torchbearer: A model training library for researchers using PyTorch.
63. [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl): Reinforcement Learning with Model-Agnostic Meta-Learning in Pytorch. 
64. [NALU](https://github.com/bharathgs/NALU): Basic pytorch implementation of NAC/NALU from Neural Arithmetic Logic Units paper by trask et.al arxiv.org/pdf/1808.00508.pdf
66. [QuCumber](https://github.com/PIQuIL/QuCumber): Neural Network Many-Body Wavefunction Reconstruction
67. [magnet](https://github.com/MagNet-DL/magnet): Deep Learning Projects that Build Themselves http://magnet-dl.readthedocs.io/
68. [opencv_transforms](https://github.com/jbohnslav/opencv_transforms): OpenCV implementation of Torchvision's image augmentations
69. [fastai](https://github.com/fastai/fastai): The fast.ai deep learning library, lessons, and tutorials
70. [pytorch-dense-correspondence](https://github.com/RobotLocomotion/pytorch-dense-correspondence): Code for "Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation" arxiv.org/pdf/1806.08756.pdf
71. [colorization-pytorch](https://github.com/richzhang/colorization-pytorch): PyTorch reimplementation of Interactive Deep Colorization richzhang.github.io/ideepcolor
72. [beauty-net](https://github.com/cms-flash/beauty-net): A simple, flexible, and extensible template for PyTorch. It's beautiful.
73. [OpenChem](https://github.com/Mariewelt/OpenChem): OpenChem: Deep Learning toolkit for Computational Chemistry and Drug Design Research mariewelt.github.io/OpenChem 
74. [torchani](https://github.com/aiqm/torchani): Accurate Neural Network Potential on PyTorch aiqm.github.io/torchani
75. [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS): A PyTorch implementation of L-BFGS.
76. [gpytorch](https://github.com/cornellius-gp/gpytorch): A highly efficient and modular implementation of Gaussian Processes in PyTorch.
77. [hessian](https://github.com/mariogeiger/hessian): hessian in pytorch. 
78. [vel](https://github.com/MillionIntegrals/vel): Velocity in deep-learning research.
79. [nonechucks](https://github.com/msamogh/nonechucks): Skip bad items in your PyTorch DataLoader, use Transforms as Filters, and more!
80. [torchstat](https://github.com/Swall0w/torchstat): Model analyzer in PyTorch.
81. [QNNPACK](https://github.com/pytorch/QNNPACK): Quantized Neural Network PACKage - mobile-optimized implementation of quantized neural network operators.
82. [torchdiffeq](https://github.com/rtqichen/torchdiffeq): Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation.
83. [redner](https://github.com/BachiLi/redner): A differentiable Monte Carlo path tracer
84. [pixyz](https://github.com/masa-su/pixyz): a library for developing deep generative models in a more concise, intuitive and extendable way. 
85. [euclidesdb](https://github.com/perone/euclidesdb): A multi-model machine learning feature embedding database http://euclidesdb.readthedocs.io 
86. [pytorch2keras](https://github.com/nerox8664/pytorch2keras): Convert PyTorch dynamic graph to Keras model.
87. [salad](https://github.com/domainadaptation/salad): Semi-Supervised Learning and Domain Adaptation.
88. [netharn](https://github.com/Erotemic/netharn): Parameterized fit and prediction harnesses for pytorch.
89. [dgl](https://github.com/dmlc/dgl): Python package built to ease deep learning on graph, on top of existing DL frameworks. http://dgl.ai. 
90. [gandissect](https://github.com/CSAILVision/gandissect): Pytorch-based tools for visualizing and understanding the neurons of a GAN. gandissect.csail.mit.edu 
91. [delira](https://github.com/justusschock/delira): Lightweight framework for fast prototyping and training deep neural networks in medical imaging delira.rtfd.io
92. [mushroom](https://github.com/AIRLab-POLIMI/mushroom): Python library for Reinforcement Learning experiments.
93. [Xlearn](https://github.com/thuml/Xlearn): Transfer Learning Library
94. [geoopt](https://github.com/ferrine/geoopt): Riemannian Adaptive Optimization Methods with pytorch optim
95. [vegans](https://github.com/unit8co/vegans): A library providing various existing GANs in PyTorch.
96. [torchgeometry](https://github.com/arraiyopensource/torchgeometry): TGM: PyTorch Geometry
97. [AdverTorch](https://github.com/BorealisAI/advertorch): A Toolbox for Adversarial Robustness (attack/defense/training) Research
98. [AdaBound](https://github.com/Luolc/AdaBound): An optimizer that trains as fast as Adam and as good as SGD.a
99. [fenchel-young-losses](https://github.com/mblondel/fenchel-young-losses): Probabilistic classification in PyTorch/TensorFlow/scikit-learn with Fenchel-Young losses
100. [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter): Count the FLOPs of your PyTorch model.
101. [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
102. [Catalyst](https://github.com/catalyst-team/catalyst): High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
103. [Ax](https://github.com/facebook/Ax): Adaptive Experimentation Platform
104. [pywick](https://github.com/achaiah/pywick): High-level batteries-included neural network training library for Pytorch
105. [torchgpipe](https://github.com/kakaobrain/torchgpipe): A GPipe implementation in PyTorch torchgpipe.readthedocs.io
106. [hub](https://github.com/pytorch/hub): Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.
107. [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning): Rapid research framework for Pytorch. The researcher's version of keras.
108. [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
109. [tensorwatch](https://github.com/microsoft/tensorwatch): Debugging, monitoring and visualization for Deep Learning and Reinforcement Learning from Microsoft Research.
110. [wavetorch](https://github.com/fancompute/wavetorch): Numerically solving and backpropagating through the wave equation arxiv.org/abs/1904.12831
111. [diffdist](https://github.com/ag14774/diffdist): diffdist is a python library for pytorch. It extends the default functionality of torch.autograd and adds support for differentiable communication between processes. 
112. [torchprof](https://github.com/awwong1/torchprof): A minimal dependency library for layer-by-layer profiling of Pytorch models.
113. [osqpth](https://github.com/oxfordcontrol/osqpth): The differentiable OSQP solver layer for PyTorch. 
114. [mctorch](https://github.com/mctorch/mctorch): A manifold optimization library for deep learning. 
115. [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings): Efficient PyTorch Hessian eigendecomposition using the Hessian-vector product and stochastic power iteration. 
116. [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine): Minkowski Engine is an auto-diff library for generalized sparse convolutions and high-dimensional sparse tensors.
117. [pytorch-cpp-rl](https://github.com/Omegastick/pytorch-cpp-rl): PyTorch C++ Reinforcement Learning
118. [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt): PyTorch extensions for fast R&D prototyping and Kaggle farming
119. [argus-tensor-stream](https://github.com/Fonbet/argus-tensor-stream): A library for real-time video stream decoding to CUDA memory tensorstream.argus-ai.com
120. [macarico](https://github.com/hal3/macarico): learning to search in pytorch
121. [rlpyt](https://github.com/astooke/rlpyt): Reinforcement Learning in PyTorch
122. [pywarm](https://github.com/blue-season/pywarm): A cleaner way to build neural networks for PyTorch. blue-season.github.io/pywarm
123. [learn2learn](https://github.com/learnables/learn2learn): PyTorch Meta-learning Framework for Researchers http://learn2learn.net
124. [torchbeast](https://github.com/facebookresearch/torchbeast): A PyTorch Platform for Distributed RL
125. [higher](https://github.com/facebookresearch/higher): higher is a pytorch library allowing users to obtain higher order gradients over losses spanning training loops rather than individual training steps.
126. [Torchelie](https://github.com/Vermeille/Torchelie/): Torchélie is a set of utility functions, layers, losses, models, trainers and other things for PyTorch. torchelie.readthedocs.org 
127. [CrypTen](https://github.com/facebookresearch/CrypTen): CrypTen is a Privacy Preserving Machine Learning framework written using PyTorch that allows researchers and developers to train models using encrypted data. CrypTen currently supports Secure multi-party computation as its encryption mechanism.
128. [cvxpylayers](https://github.com/cvxgrp/cvxpylayers): cvxpylayers is a Python library for constructing differentiable convex optimization layers in PyTorch
129. [RepDistiller](https://github.com/HobbitLong/RepDistiller): Contrastive Representation Distillation (CRD), and benchmark of recent knowledge distillation methods
130. [kaolin](https://github.com/NVIDIAGameWorks/kaolin): PyTorch library aimed at accelerating 3D deep learning research
131. [PySNN](https://github.com/BasBuller/PySNN): Efficient Spiking Neural Network framework, built on top of PyTorch for GPU acceleration.
132. [sparktorch](https://github.com/dmmiller612/sparktorch): Train and run Pytorch models on Apache Spark.
133. [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning): The easiest way to use metric learning in your application. Modular, flexible, and extensible. Written in PyTorch.
134. [autonomous-learning-library](https://github.com/cpnota/autonomous-learning-library): A PyTorch library for building deep reinforcement learning agents.
135. [flambe](https://github.com/asappresearch/flambe): An ML framework to accelerate research and its path to production. flambe.ai
136. [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer): Collections of modern optimization algorithms for PyTorch, includes: AccSGD, AdaBound, AdaMod, DiffGrad, Lamb, RAdam, RAdam, Yogi.
137. [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE): A Collection of Variational Autoencoders (VAE) in PyTorch.
138. [ray](https://github.com/ray-project/ray): A fast and simple framework for building and running distributed applications. Ray is packaged with RLlib, a scalable reinforcement learning library, and Tune, a scalable hyperparameter tuning library. ray.io
139. [Pytorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal): A temporal extension library for PyTorch Geometric 
140. [Poutyne](https://github.com/GRAAL-Research/poutyne): A Keras-like framework for PyTorch that handles much of the boilerplating code needed to train neural networks.
141. [Pytorch-Toolbox](https://github.com/PistonY/torch-toolbox): This is toolbox project for Pytorch. Aiming to make you write Pytorch code more easier, readable and concise.
142. [Pytorch-contrib](https://github.com/pytorch/contrib): It contains reviewed implementations of ideas from recent machine learning papers.
143. [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): It contains an op-for-op PyTorch reimplementation of EfficientNet, along with pre-trained models and examples.
144. [PyTorch/XLA](https://github.com/pytorch/xla): PyTorch/XLA is a Python package that uses the XLA deep learning compiler to connect the PyTorch deep learning framework and Cloud TPUs.
145. [webdataset](https://github.com/tmbdev/webdataset): WebDataset is a PyTorch Dataset (IterableDataset) implementation providing efficient access to datasets stored in POSIX tar archives.
146. [volksdep](https://github.com/Media-Smart/volksdep): volksdep is an open-source toolbox for deploying and accelerating PyTorch, Onnx and Tensorflow models with TensorRT.
147. [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN): StudioGAN is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze a new idea.
148. [torchdrift](https://github.com/torchdrift/torchdrift/): drift detection library
149. [accelerate](https://github.com/huggingface/accelerate) : A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision 
150. [lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers):  Flexible interface for high-performance research using SOTA Transformers leveraging Pytorch Lightning, Transformers, and Hydra. 
151. [Flower](https://flower.dev/) A unified approach to federated learning, analytics, and evaluation. It allows to federated any machine learning workload.
152. [lightning-flash](https://github.com/PyTorchLightning/lightning-flash): Flash is a collection of tasks for fast prototyping, baselining and fine-tuning scalable Deep Learning models, built on PyTorch Lightning.
153. [Pytorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed): A signed and directed extension library for PyTorch Geometric. 
154. [Koila](https://github.com/rentruewang/koila): A simple wrapper around pytorch that prevents CUDA out of memory issues.
155. [Renate](https://github.com/awslabs/renate): A library for real-world continual learning.
156. [ANEE](https://github.com/abkmystery/ANEE) – Adaptive Neural Execution Engine for PyTorch transformers. Provides per-token dynamic layer skipping, profiler-based gating, and KV-cache-safe sparse inference.


## Tutorials, books, & examples

1. **[Practical Pytorch](https://github.com/spro/practical-pytorch)**: Tutorials explaining different RNN models
2. [DeepLearningForNLPInPytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html): An IPython Notebook tutorial on deep learning, with an emphasis on Natural Language Processing. 
3. [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial): tutorial for researchers to learn deep learning with pytorch.
4.  [pytorch-exercises](https://github.com/keon/pytorch-exercises): pytorch-exercises collection. 
5.  [pytorch tutorials](https://github.com/pytorch/tutorials): Various pytorch tutorials. 
6.  [pytorch examples

