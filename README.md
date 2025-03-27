# Image Processing with Gaussian Filter, Otsu Thresholding, and Sobel Edge Detection

## Overview
This project processes images using advanced image processing techniques. The program is written in C and utilizes OpenMP for parallel computing. It implements Gaussian filtering, Otsu's thresholding, and Sobel edge detection using both sequential and parallel techniques.

## Features
- Load images in various formats (JPEG, PNG, etc.).
- Apply:
  - **Gaussian Filter**: Smooths images by reducing noise.
  - **Otsu Thresholding**: Converts grayscale images to binary using an optimal threshold.
  - **Sobel Edge Detection**: Detects edges using the Sobel operator.
- Execute these operations sequentially or in parallel using OpenMP.
- Save processed images in the desired format.
- Interactive command-line interface for selecting processing techniques.

## Prerequisites
To compile and run this project, you need:
- A C compiler supporting OpenMP (GCC, Clang, etc.)
- `stb_image.h` and `stb_image_write.h` (included in the project)

## Installation and Compilation
1. Clone the repository or copy the source files.
2. Compile the C program with OpenMP support:
   ```sh
   gcc medical_image_processing_color_interactive.c -o image_processor -lm -fopenmp
   ```
3. Run the program:
   ```sh
   ./image_processor
   ```

## Usage
- The program will prompt for an input image file.
- Users can select the desired processing technique:
  - **Gaussian Filtering**
  - **Otsu Thresholding**
  - **Sobel Edge Detection**
- Users can choose between sequential or parallel execution.
- The processed image is saved with a specified output filename.

## Dependencies
This project uses the following libraries:
- `stb_image.h`: For image loading.
- `stb_image_write.h`: For saving the processed image.
- `OpenMP`: For parallel processing.

## License
This project is open-source and follows the MIT License.

## Acknowledgments
- Sean Barrett for the `stb` image library.
- Open-source contributors for image processing and parallel computing techniques.

## Contact
For any issues or improvements, feel free to contribute or report an issue!

