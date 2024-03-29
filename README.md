# Variational Bayesian GAN for MNIST Generation
In this repository, we implement our proposed Varational Bayesian GAN (VBGAN) and VBGAN with Wasserstein metric based on auto-encoder based generator and train the model with adversarial learning. The latter extend the Jensen Shannon divergence between data distribution and generating distribution to geometric distance to make the model more robust and genearate better result. To quantify the result of our proposed, we also train our model in semi-supervised learning and the accuracy results show that our model did learning some information from adversarial learning process.


Bayesian GAN credit to https://github.com/vasiloglou/mltrain-nips-2017/blob/master/ben_athiwaratkun/pytorch-bayesgan/Bayesian%20GAN%20in%20PyTorch.ipynb

Bayes by Backbrop credit to https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac

* Our Model Architecture

<p align="center">
  <img src="figures/Model_slide.PNG" width="450">
	<em>VBGAN</em>
</p>
<p align="center">
  <img src="figures/Model_slide_w.PNG" width="450">
	<em>VBGAN with Wasserstein metric</em>
</p>

## Setting
- Framework:
    - Pytorch 0.4.0
- Hardware:
	- CPU: Intel Core i7-2600 @3.40 GHz
	- RAM: 20 GB DDR4-2400
	- GPU: GeForce GTX 980

## Result of sampling
| <img src="figures/VBGAN_sam.png" width="400"> |
| :------------------------------------------------: |
| VBGAN                                   |

| <img src="figures/VBGAN_wasserstein_sam.png" width="400/"> |
| :--------------------------------------------------: |
| VBGAN_w                                           |

## Test accuracy
| <img src="figures/4000.png" width="400"> |
| :-----------------------------------------: |
| Test accuracy of our proposed and Bayesian GAN  |

