# Variational Bayesian GAN for MNIST Generation
In this repositor, we implement our proposed VBGAN and VBGAN with Wasserstein metric based on auto-encoder based generator and train the model with adversarial learning. Further, we extend the Jensen Shannon divergence between data distribution and generating distribution to geometric distance to make the model more robust and genearate better result. To quantify the result of our proposed, we also train our model with semi-supervised learning to show that our model did learning some information from training data.

<p align="center">
  <img src="figures/Model_slide.PNG" width="450">
  <img src="figures/Model_slide_w.PNG" width="450">
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

