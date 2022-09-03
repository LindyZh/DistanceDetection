# Distance Detection for MTA dataset
![distance detection](https://github.com/LindyZh/DistanceDetection/blob/main/img/demo.jpg)
## Dataset
All dataset used for distance detection training and testing are acquired from the [MTA (multi camera track auto) dataset](https://github.com/schuar-iosb/mta-dataset)\

## Distance Detection Method
We have explored and tested multiple methods of distance detection non-machine-learning methods, including:
 - assuming Cartesian coordinates for the ground plane and calculate the euclidean distance.
 - assuming Polar coordinates for the ground plane and calculate the euclidean distance
 - using space transformation to transform the 3d camera plane (pre-configured using the existing camera data) to a 2d one.

The final method that we employed is the Cartesian method as it provides the best accuracy among all possibilities.

 
