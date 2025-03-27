# Image Processing with Color Manipulation

## Overview
This project processes normal images using color manipulation techniques. The program is written in C and utilizes the `stb_image.h` and `stb_image_write.h` libraries for reading and writing images. The core logic is implemented in `medical_image_processing_color_interactive.c`.

## Features
- Load images in various formats (JPEG, PNG, etc.).
- Apply color transformations and modifications.
- Save processed images in the desired format.
- Interactive command-line interface for color adjustments.

## Prerequisites
To compile and run this project, you need:
- A C compiler (GCC, Clang, etc.)
- `stb_image.h` and `stb_image_write.h` (included in the project)

## Installation and Compilation
1. Clone the repository or copy the source files.
2. Compile the C program using:
   ```sh
   gcc medical_image_processing_color_interactive.c -o image_processor -lm
   ```
3. Run the program:
   ```sh
   ./image_processor
   ```

## Usage
- The program will prompt for an input image file.
- Users can specify color processing parameters interactively.
- The processed image is saved with a specified output filename.

## Dependencies
This project uses the following libraries:
- `stb_image.h`: For image loading.
- `stb_image_write.h`: For saving the processed image.

## License
This project is open-source and follows the MIT License.

## Acknowledgments
- Sean Barrett for the `stb` image library.
- Open-source contributors for image processing techniques.

## Contact
For any issues or improvements, feel free to contribute or report an issue!

