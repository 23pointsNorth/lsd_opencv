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
cd build/
cmake ..
make


Use
-----
To use the OpenCV LSD, create and LSD object, calling detect with the input image and a vector of lines.


Testing
-----
To test the difference between the standard and the converted algorithm, run

./visual_test ./../images/any-image

to test the algorithm in specific cases, use the 

./accuracy_test

It will run the code against a set of predefined cases.
