# sky360lib
C++ Library/Apps for background subtraction and other algorithms for sky360 project

# ViBe
Right now the ViBe BS is the only one implemented, you can learn more about ViBe from:

http://www.telecom.ulg.ac.be/publi/publications/barnich/Barnich2011ViBe/index.html

# Getting and Building

* You need a development environment to build with:
  - Build tools (gcc, cmake)
  - OpenCV > 4.0
  
* Open a terminal:
  ## Building OpenCV
  - sudo apt install build-essential cmake -y
  - create a new dir:
    - mkdir sky360
    - cd sky360
  - git clone https://github.com/opencv/opencv.git
  - mkdir build
  - cd build
  - cmake ..
  - cmake --build .
  - sudo cmake --install .
  
  ## Building the library/demo
  - go to the sky360 directory
  - git clone https://github.com/Sky360-Repository/sky360lib.git
  - cd embedded-bgsub
  - mkdir build
  - cd build
  - cmake ..
  - cmake --build .
  
  ## Running the demo
  - go to the sky360 directory
  - cd build/bin
  - sky360lib_demo 0
    - The number is the camera number, you might need to change it to 1, 2
