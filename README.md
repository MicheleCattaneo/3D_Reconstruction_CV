# Computer Vision project: 3D reconstruction from an image

<img src='house1.png' width="400px" />

## Part 1: Rectifying the house front that contains the red window and the blue door affinely

Apply Canny edge detection on the image:

<img src='./figures/5 canny_ours.png' width="400px" />

Use the Hough transform on the detected edges to find lines on the front of the house. Then two set of lines that should be parallel in the real scene are selected.

<div style="display: flex;">
  <img src="./figures/front_lines.png" alt="Image 1" width="400" height="300">
  <img src="./figures/parallel_lines.png" width="400" height="300">
</div>
