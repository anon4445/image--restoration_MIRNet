# Image Processing with Multi-scale Image Restoration Network 

## Features
- Four Powerful Image Processing Tasks: Choose from real denoising, super-resolution, contrast enhancement, and lowlight enhancement.
- Interactive GUI: Powered by Gooey, a user-friendly interface to select and process images.
- Batch Processing: Ability to process an entire directory of images.
## Supported Tasks
- Real Denoising: This task aims to remove unwanted noise from images, enhancing clarity. Pre-trained weights for this task are saved in the model directory under the filename real_denoising.pth.
- Super Resolution: Super resolution enhances the resolution of images, making them sharper. It is particularly useful for upscaling low-resolution images to higher resolutions without introducing blurriness. Pre-trained weights for this task are saved in the model directory under the filename sr_x2.pth. This task specifically doubles the resolution (scale parameter is set to 2).
- Contrast Enhancement: This task enhances the contrast of images, making them more vibrant and clearer, particularly helpful for images that are washed out or faded. Pre-trained weights for this task are saved in the model directory under the filename enhancement_fivek.pth.
- Lowlight Enhancement: Lowlight enhancement is aimed at improving images that are taken in poor light conditions. It brightens and clarifies such images, revealing more details. Pre-trained weights for this task are saved in the model directory under the filename enhancement_lol.pth.

For each of these tasks, the user can select the desired task via the GUI. Once selected, the program loads the appropriate pre-trained weights, processes the images from the specified input directory, and saves the results to the output directory. 
