# XSurv: Merging-Diverging Hybrid Transformer Networks for Survival Prediction
We propose a merging-diverging learning framework for survival prediction from multi-modality images. This framework has a merging encoder to fuse multi-modality information and a diverging decoder to extract region-specific information. In the merging encoder, we propose a Hybrid Parallel Cross-Attention (HPCA) block to effectively fuse multi-modality features via parallel convolutional layers and cross-attention transformers. In the diverging decoder, we propose a Region-specific Attention Gate (RAG) block to screen out the features related to lesion regions. Our framework is demonstrated on survival prediction from PET-CT images in Head and Neck (H&N) cancer, by designing an X-shape merging-diverging hybrid trans-former network (named XSurv). Our XSurv combines the complementary information in PET and CT images and extracts the region-specific prognostic information in Primary Tumor (PT) and Metastatic Lymph Node (MLN) regions. Extensive experiments on the public dataset of HEad and neCK TumOR segmentation and outcome prediction challenge (HECKTOR 2022) demonstrate that our XSurv outperforms state-of-the-art survival prediction methods.  
**For more details, please refer to our paper. [[arXiv](https://arxiv.org/abs/2307.03427)]**

## Architecture
![architecture](https://github.com/MungoMeng/Survival-XSurv/blob/master/Figure/architecture.png)

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Lei Bi, Michael Fulham, Dagan Feng, Jinman Kim, "Merging-Diverging Hybrid Transformer Networks for Survival Prediction in Head and Neck Cancer," International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 400-410, 2023, doi: 10.1007/978-3-031-43987-2_39. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_39)] [[arXiv](https://arxiv.org/abs/2307.03427)]**
