#### Manual install
|Dependencies  |
| ------------- |
| python==3.5.2|
| pytorch==0.4.1|
| cuda==8.0|
| torchvision==0.2.1|
| torchcontrib==0.0.2|
| matplotlib==3.0.1|
| scikit-learn==0.20.0|
| tqdm==4.28.1|
| numpy==1.15.3|

### Usage

You can find an example script to run the FMCmatch on CIFAR-10 with 1000 labeled samples in [RunScripts_SOTA_CNN_1000.sh]
Execute the script from the corresponding folder to train the model.

### Parameters details
The most relevant parameters are the following:
* --labeled_samples: Number of labeled samples 
* --epoch: Number of epochs of training
* --M: Epochs where the learning rate is divided by 10
* --network: Network architecture ("MT\_Net", "WRN28\_2\_wn", or "resnet18\_wndrop")
* --DA: Type of data augmentation ("standard" or "jitter")

To run the experiments download the corresponding dataset in the folder ./CIFAR10/data or ./CIFAR100/data or ./SVHN or ./miniimagenet.



### Acknowledgements

We would like to thank [1] (Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning) and [2] (FMix:Understanding and Enhancing Mixed Sample Data Augmentation) for their code sharing.

[1] Eric Arazo, Diego Ortego, Paul Albert, Noel E. O'Connor, Kevin McGuinness, Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning, arXiv:1908.02983, 2019

[2] Harris Ethan, Marcu Antonia, Painter Matthew, Niranjan Mahesan, Hare Adam, FMix:Understanding and Enhancing Mixed Sample Data Augmentation, arXiv:2002.12047, 2020



