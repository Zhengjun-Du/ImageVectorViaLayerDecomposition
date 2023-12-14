## Image vectorization and editing via linear gradient layer decomposition

![](https://github.com/Zhengjun-Du/ImageVectorViaLayerDecomposition/blob/main/teaser.png)

This is the source code of the paper: **Image vectorization and editing via linear gradient layer decomposition**, authors: Zheng-Jun Du,  Liang-Fu Kang, Jianchao Tan, Yotam Gingold, Kun Xu. Please fell free to contact us if you have any questions, email: dzj@qhu.edu.cn

### Requirements

Windows 10  
Microsoft Visual Studio 2019  
OpenCV 4.1.2 or higher version  
Nlopt 2.4.2 (included in "ImageVectorization/ThirdParty")  
autodiff (included in "ImageVectorization/ThirdParty")  
Python 3.7 (used to generate vector graph with .svg format) 

### Directories

1. Data: we provide 10 examples in this directory
2. ImageVectorization： main program for layer deomposition
3. ProcessRegionSegImg: preprocess the region segmentation input
4. Gen_svg_script: contain a python script to generate the vector graph 

### Usage

1. Open "./ImageVectorization" -> click ImageVectorization.sln -> run "main.cpp". It will automatically decomposes the input image into a set of layers and savea the results in "data/xxx/results".  

   If you want to test your examples, you could use the "ProcessRegionSegImg" project to preprocess your segmentation images first.

2. Run "Gen_svg_script/main.py" to generate the vector graph with .svg format.

### Reference

[1] Du Z J, Kang L F, Tan J, et al. Image vectorization and editing via linear gradient layer decomposition[J]. ACM Transactions on Graphics (TOG), 2023, 42(4): 1-13.
