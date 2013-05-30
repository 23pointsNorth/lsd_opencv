LSD OpenCV
==========

Introduction
-----
A port of Line Segment Detector(LSD) to use OpenCV structures, as part of the GSoC 2013 program. 
The original code and paper, developed by Rafael Grompone von Gioi <grompone@gmail.com>, can be found at http://www.ipol.im/pub/art/2012/gjmr-lsd/ .


Files
-----
The source files are separated in the following directories:
	* src/ 		- main files that are build to different shared libs
	* examples/ - contains a variety of examples on how to use the code
	* tests/	- different ways to test the performance

Extra
	* docs/		- documentation from the original code
	* images/	- contains images for easy use and tests


Compile
-----
On linux, navigate to the lsd_1.6 directory and execute
mkdir build
cd build
cmake ..
make


Todo:
-----
	*	Start converting lsd.cpp to lsd_opencv.cpp
	*