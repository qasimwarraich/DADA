# Exploiting Semantics and Cycle Association for Domain-Adaptive Semantic Segmentation

Authors: Rohit Kaushik, Qasim Warraich

[[Link to this repository](https://github.com/qasimwarraich/DADA/tree/final)]

## Abstract 
 Performing domain adaptive semantic segmentation is one of the core vision problems as it facilitates automated visual scene understanding without the need for expensive and labor-intensive annotations. However, due to the inevitable domain shift problem, the model performance significantly degrades when tested on unseen target samples. Research into how best to mitigate the domain shift problem is a highly active research area. The majority of the existing approaches exploit adversarial learning to minimize the distribution discrepancy. Recently contrastive learning has been proposed as an alternative to tackle domain adaptation. In this work, we further investigate this new research direction. More specifically, we study how a contrastive objective, based on pixel-level cycle association, could help learn better a generalisable representation. The idea here is to bring the feature embeddings closer for the visually similar pixels and push the dissimilar pixels further apart. Pixel similarity is measured using a cycle-association technique that outputs pairs of pixels (from source and target images) having the least distance in the embedding space. Our experimental results on the challenging SYNTHIA to Cityscapes benchmark demonstrate that pixel-level contrastive learning is a promising approach that improves semantic segmentation results. 

 ## This repository
This repository is a fork of the [DADA](https://github.com/valeoai/DADA) work and we refer you to the original DADA [readme](https://github.com/valeoai/DADA/blob/master/README.md) for installation instructions. However, we only really rely upon the DADA dataloader and have built our own contrastive module on top of it. In order to utilise this module please set `CONTRASTIVE_LEARNING = True` in the `config.py` file. 

Our contrastive module is a from scratch implementation of the Pixel Level Cycle Association work described by [Kang et al.](https://papers.nips.cc/paper/2020/hash/243be2818a23c980ad664f30f48e5d19-Abstract.html). This module employs pixel wise similarity to bridge the domain gap by associating visually similar pixels and enforcing a semantic class consistency in the source domain. For a more detailed view into this approach we refer you to our paper. 

## Navigating this repository

The final state of our project can be found on the `final` branch [[Link](https://github.com/qasimwarraich/DADA/tree/final)].

Most of the action takes place in the `DADA/dada/` subdirectory. In the `domain_adaptation/` directory you can find our implementation in the `contrastive_learning.py` file. Associated training scripts and scripts containing the various loss functions can also be found in this directory. In the `scripts/` directory you can find our script for producing visualisations as shown in our paper in addition to various other batch scripts for running the training on a `SLURM` based GPU cluster. Additionally under the `tests/` directory you can find a unit test we wrote for validating our implementation of Pixel Level Cycle Association. 



