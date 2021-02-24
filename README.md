<!-- ### Deep learning project seed -->
<!-- Use this seed to start new deep learning / ML projects. -->

<!-- - Built in setup.py -->
<!-- - Built in requirements -->
<!-- - Examples with MNIST -->
<!-- - Badges -->
<!-- - Bibtex -->

<!-- #### Goals   -->
<!-- The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.    -->

<!-- ### DELETE EVERYTHING ABOVE FOR YOUR PROJECT   -->
 
<!-- --- -->

<div align="center">
 
# Automated landmarking for palatal shape analysis     

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


<!--  
Conference   
-->   
</div>
 
## Description   
This study is the first to develop geometric-deep-learning-based automatic landmarking of the palate. The network automatically places seven palatal landmarks on the canines, first molars, and on the palatal   midline of 3D surface scans of maxillary dental casts. The networkemploys a PointNet++-based network architecture to hierarchically learn features from point-clouds representing the surface of each cast. The network is trained from manual landmark annotations  on 740 dental casts and evaluated  on 106 dental casts.

## How to run (TODO)  
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

### Citation (TODO)   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
