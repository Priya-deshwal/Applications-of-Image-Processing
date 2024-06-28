# Applications-of-Image-Processing
## About
  In this project I have developed algorithms to segment out regions in images and produce mask images, and to enhance the quality 
  of two images when mergerd (flash and non-flash images) and to detect material by analysing spectrogram obtained by sound of the 
  material and implemented hough transformation and k- mean clusturing and algorithms like DCT to detect counterfeit images.
## Technologies required
 * python 3.10
 * anaconda
 * opencv-python==4.8.0.76
 * scikit-image==0.21.0
 * librosa==0.10.1
## Environment Setup
  For Ubantu/Mac install python3.10 and install module venv
  ```
  $ sudo apt install python3 .10 - venv
  ```
  create virtual environment
  ```
  $ python3 -m venv ee604
  ```
  activate environment and install required modules:
  ```
  $ source ee604 / bin / activate
  $ pip install -r requirements . txt
  ```
  for windows, install anaconda and use following command to setup and activate environment and to install required modules:
  ```
  $ conda create -n ee604 python =3.10 anaconda
  $ conda activate ee604
  $ conda install -c conda - forge opencv
  $ conda install -c conda - forge librosa
  $ conda install -c anaconda scikit - image
  ```
