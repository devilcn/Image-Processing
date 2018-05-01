# Mid-term

- Computer graphic: the process of creating images from abstract model. For e.g. there's a list/Matrix of points, each point correspond to each pixel of the image.<br>
- Image processing: Starting with an exsiting image and using different methods to refine it.

Gray-Scale Image:<br>
for e.g. RGB image (3 channels), if n bits/cell or n bits/pixel, which means n/3 bits per channel, as the result the image can show `math.pow(2,n)` types of color, and `math.pow(2,n/3)` types of shades of gray

CLT (color look-up table): Increase the nums of option of color<br>
for e.g. 9 bits for each cell/pixel, so there can be `math.pow(2,9)` types of color in the image, but if the size of each row of CLT is `math.pow(2,12)`, then the option of color is `math.pow(2,12)`

Image histogram and info content

Linear contrast enhancement method and its implementation:<br>
If the image suffers from low-contrast, we can use contrast enhancement, but we will miss info of the image.

Conversion of color to gray-scale:<br>

luminosity:

```python
            avglist = img[i,j]
            
            avg = int(avglist[0]*0.299 + avglist[1]*0.587 + avglist[2]*0.114)
            
            img[i,j] = [avg,avg,avg]
```
***
### Conversion of gray-scale to B&W:<br>
0. Slicing: When multiple peeks appears in a histogram, use multiple t, compute each of them then combine the results.
1. Simple thresholding: `img[i,j,0] > t -> White, else -> Black`<br>
2. P-Tile: Top x% pixel be white, others be black<br>
3. Iterative threshold selection: <br>

```
t = random.randint(0,255)
using t as thresholding to convert the image to B&W
compuing the avg value of each part a1,a2
newt = (a1+a2)/2
if abs(newt-t) < 1:
	return newt
else:
	using t as thresholding to convert the image to B&W and computing a1,a2 until (newt-t)<1

```

***

#### Low-pass filter:<br>

- All values in masks are positive

- The sum of all the values equals to 1

- The boundary content is reduced by using a low-pass filter

- As the size of mask grows, more smoothing will take place

#### High-pass filter:<br>

- A mask have positive and negative values

- The sum of all values in a mask is equal to 0

- The boundary content is increased by a high-pass mask

- As the size of mask grows, more boundary content is increased

***

### Noise Reduction:<br>

- averaging: NS5

- Median: NS9

Smoothing using thresholding: withour lossing boundary.

***

Component labeling and applications:

***
### Image sharpening:<br>

0. gradient operators<br>

1. 1st and 2nd order derivatives: 1st order: gradient operators; 2nd order: Laplacian operator<br>

2. Robert's operator: `[0,-1,1,0]+[-1,0,0,1]`<br>
3. sharpening by thresholding: base on Robert's<br>
4. Laplacian operator<br>
`ori1 = [0,1,0,1,-4,1,0,1,0], ori2 = [1,0,1,0,-4,0,1,0,1], new = [1,1,1,1,-8,1,1,1,1]`<br>
5. Prewitt operator: child<br>
`r1 = [-1,0,1,-1,0,1,-1,0,1], r2 = [1,1,1,0,0,0,-1,-1,-1], result = math.sqrt(r1**2+r2**2)`<br>
6. Kirsh operator: parent<br>
`r3 = [0,1,1,-1,0,1,-1,-1,0], r4 = [1,1,0,1,0,-1,0,-1,-1], result = max(r1,r2,r3,r4)`<br>
7. Sobel operator:<br>
 `x = [1,2,1,0,0,0,-1,-2,-1], y = [1,0,-1,2,0,-2,1,0,-1], result = math.sqrt(x**2+y**2)`<br>
8. conversion of high-pass filtered images into B&W
***

### Parallel Processing & HPC

- ALU: Arithmetic Logic Unit<br>
- MCU: Master Control Unit<br>

####Grosch's Law <br>
To sell a computer for twice as much, it must be four times faster.<br>

#### Von Neumann's Design Rule <br>

Multiple Processors performance = N\*T - V <br>
 - N: No. of processors <br>
 - T: No. of Instructions per second for each <br>
 - V: Communication Overhead <br>

#### Amdahl's Law <br>
 
 `Speedup = T(1) / T(N)` <br>
 Speedup is the theoretical speedup of the execution of the whole task. <br>
 - B: The serial part of the code can be computed in the equal to `BT(1)` <br>
 - (1-B): The parellel part of the code can be computed in the equal to `(1-B)T(1) / N` <br>
 `Speedup = T(1) / (T(1)B + T(1)(1-B)/N) = N / (BN + (1-B))`  <-- Pessimistic View <br>

 Eg. Assume N = 10, B = 0.67, (1-B) = 0.33 <br>
 ==> `Speedup = 1/(0.67+(0.33/10)) = 1.42` <-- This is not ideal<br>
#### Gustafson's Law <br>
 Consider B & N are not independent
 `Speedup = T(1)/T(N)`<br>
 If N = 1, `T(1) = B + (1-B)N`<br>
 If N > 1, `T(N) = B + (1-B) = 1` <br>
 `Speedup = [B + (1-B)N]/1`,  `Speedup = N - B(N-1) or N + B(1-N)`  <-- Optimistic View<br>
 Eg. Assume N = 10, B = 0.67, (1-B) = 0.33 <br>
 ==> `Speedup = 10 - 0.67(9) = 3.97 ~= 4` <-- Expected the speedup of 4.<br>
 
 # Final
 
 ## Magification

The goal: reveal temporal variations in videos that are difficult to see by naked eye.

The method: Called Eulerian Video Magnification.

- takes a standard video sequence as input
- Applies spatial decomposition
- Followed by temporal filtering to the frames
- The resulting signal is then amplified to reveal hidden information.


## Object Recognition

YOLO utilizes entire images during training and testing utilizes a single convolutional neural network to simultaneously predict multiple bounding boxes and probabilities associated probabilities.
Limitations: YOLO imposes a constraint on the number of boxes that can predicted per cell, 2, and only one class prediction per cell, so it struggles to detect small and nearby objects


## A Primer on Deep Learning

DL is an algorithm that is able to learn from data. DL deals with Machine Learning models that have multiple hidden layers organized in a hierarchical fashion. Eg. CNN


## Semantic Segmentation

It is about to understanding an image at pixel level, i.e. we want to assign each pixel in the image an object class.
Application: Autodrive

Mobile Real-time Video Segmentation
It allows mobile devices to do video segmentation in a real-time, precise fashion. 
Special input includes R,G,B and previous mask which accerlerates the segmentation speed. Large strides and skip connections imporoves the performance in terms of speed.


## Human Vision

There's a large difference between the image we display and image we actually perceive.

- what our eyes perceive of a scene is determined by the light rays emitted or reflected from that scene.
- when these light rays are strong enough and are within the right range of EM spectrum(300 to 700nm), the healthy eye will react to such a ray by sending an eletric signal to the brain through the optic nerve.
- Two fundamental parts: 1. Eye - functions as the biological equivalent of a camera.2. Brain - does all the complex image processing.

- we must be careful in taking our perception of an image as a gold standard; what an animal, a camera perceives in the same image may be somthing completely different.
- Our visual perception depends on what's surrounding us.


## Facial Recognition and Landmark Perdition

- Analysis phase: is always the first pass and is the most complicated. It require some computer vision algorithms running under the hood.
- Processing phase: once we have the facial landmarks done, all we have to do right now is to superpose the target filter or mask.
- given an input image of video frame, find out all present human faces and output their bounding box. The most widely used technique is a combination of Histogram of Oriented Gradients and Support Vector Machine that achieve mediocre to relatively good detection ratios given a good quality images.

How it works:

- given an input image, compute the pyramidal representation of that image which is a pyramid of multi scaled downed version of the original image.
- For each entry on the pyramid, a sliding window approach is uesd. The sliding window concept is quite simple. By looping over an image with a constant step size, small image patches pixels are extracted at different scales.
- For each patch, the algorithm makes a decision if it contains a face or not. The HOG is computed for the current window and passed to the SVM classifier for the decision to take place.


## JPEG Image Compression

(Pic to show)
Reasons:

- As a lossy compression JPEG performs very well.
- JPEG is very widely used as 'the standard' in many applications.


## Feature Detection and Extraction in Image Processing

A feature is a significant piexe of information extracted from an image which provides more detailed understanding of the image. The major goal is to extract a set of features, which maximizeds the recongnition rate with the least amount of elements. 

- Feature Detector: Finding interesting points in an image such that they remain locally invariant so that they can be detected even in a rotation or scale change. Edge detection/ corner detection/ blob detection.
- Feature Descriptor: Algorithm which takes an image and outputs feature vectors. It encodes it into a series of numbers that act like a numerical fingerprint used to fieerentiate one feature from another. SIFT/ HOG/ SURF


## Adversarial Examples in Image Classifiers

- white box attack: adversary has compolete access to model and its parameters.
- black box attack: 
1. Transferability, adversarial samples that fool model A have a huge chance of fooling a prevoiusly unseen model B.
2. Substitute model, train a new model by treating the target model's output as a training labels, then, generate adversarial samples with substitute model.
