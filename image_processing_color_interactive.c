/*
    medical_image_processing_color_interactive.c

    This program demonstrates three image processing techniques on color images:
      1. Gaussian Blur (color)
      2. Otsu Thresholding (after converting to grayscale)
      3. Sobel Edge Detection (after converting to grayscale)

    It provides both sequential and OpenMP-parallel implementations, measures their
    execution time, and writes the results to image files.

    To compile (example using GCC on Linux):
      gcc -O2 -fopenmp -lm medical_image_processing_color_interactive.c -o mip_color

    Usage:
      ./mip_color input_image.(png|jpg|jpeg)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Include stb_image and stb_image_write.
   Make sure that stb_image.h and stb_image_write.h are in your include path.
*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*---------------------------------------------------------------------------
  Helper Function: Convert RGB image to Grayscale
  Uses the standard luminosity method: gray = 0.299R + 0.587G + 0.114B
  'input'  : pointer to input image data (with 'channels' per pixel)
  'gray'   : pointer to output grayscale buffer (1 channel per pixel)
---------------------------------------------------------------------------*/
void rgb_to_grayscale(unsigned char *input, unsigned char *gray, int width, int height, int channels) {
    int total = width * height;
    for (int i = 0; i < total; i++) {
        int r = input[i * channels + 0];
        int g = input[i * channels + 1];
        int b = input[i * channels + 2];
        int y = (int)(0.299 * r + 0.587 * g + 0.114 * b);
        if (y > 255) y = 255;
        gray[i] = (unsigned char)y;
    }
}

/*---------------------------------------------------------------------------
  1. Gaussian Blurring on Color Images (5x5 kernel)
  The same 5x5 Gaussian kernel is applied to each channel.
---------------------------------------------------------------------------*/
void gaussian_blur_color_seq(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    const int kernel_size = 5;
    const int offset = kernel_size / 2;
    float kernel[5][5] = {
        { 1,  4,  6,  4, 1},
        { 4, 16, 24, 16, 4},
        { 6, 24, 36, 24, 6},
        { 4, 16, 24, 16, 4},
        { 1,  4,  6,  4, 1}
    };
    const int kernel_sum = 256;

    /* Process only the inner region.
       (You may wish to extend or handle borders separately.)
    */
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int pixel = input[((y + ky) * width + (x + kx)) * channels + c];
                        sum += pixel * kernel[ky + offset][kx + offset];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)(sum / kernel_sum);
            }
        }
    }
}

void gaussian_blur_color_par(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    const int kernel_size = 5;
    const int offset = kernel_size / 2;
    float kernel[5][5] = {
        { 1,  4,  6,  4, 1},
        { 4, 16, 24, 16, 4},
        { 6, 24, 36, 24, 6},
        { 4, 16, 24, 16, 4},
        { 1,  4,  6,  4, 1}
    };
    const int kernel_sum = 256;

    #pragma omp parallel for collapse(2)
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int pixel = input[((y + ky) * width + (x + kx)) * channels + c];
                        sum += pixel * kernel[ky + offset][kx + offset];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)(sum / kernel_sum);
            }
        }
    }
}

/*---------------------------------------------------------------------------
  2. Otsu Thresholding on Grayscale Images
  Returns the computed threshold. The output is a binary image (0 or 255).
---------------------------------------------------------------------------*/
unsigned char otsu_thresholding_seq(unsigned char* input, unsigned char* output, int width, int height) {
    int total = width * height;
    int hist[256] = {0};

    for (int i = 0; i < total; i++) {
        hist[input[i]]++;
    }

    float sum = 0;
    for (int t = 0; t < 256; t++) {
        sum += t * hist[t];
    }

    float sumB = 0;
    int wB = 0;
    int threshold = 0;
    float varMax = 0;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0)
            continue;
        int wF = total - wB;
        if (wF == 0)
            break;
        sumB += t * hist[t];
        float mB = sumB / (float)wB;
        float mF = (sum - sumB) / (float)wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }

    for (int i = 0; i < total; i++) {
        output[i] = (input[i] > threshold) ? 255 : 0;
    }
    return (unsigned char)threshold;
}

unsigned char otsu_thresholding_par(unsigned char* input, unsigned char* output, int width, int height) {
    int total = width * height;
    int hist[256] = {0};

    #pragma omp parallel
    {
        int local_hist[256] = {0};
        #pragma omp for nowait
        for (int i = 0; i < total; i++) {
            local_hist[input[i]]++;
        }
        #pragma omp critical
        {
            for (int j = 0; j < 256; j++) {
                hist[j] += local_hist[j];
            }
        }
    }

    float sum = 0;
    for (int t = 0; t < 256; t++) {
        sum += t * hist[t];
    }

    float sumB = 0;
    int wB = 0;
    int threshold = 0;
    float varMax = 0;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0)
            continue;
        int wF = total - wB;
        if (wF == 0)
            break;
        sumB += t * hist[t];
        float mB = sumB / (float)wB;
        float mF = (sum - sumB) / (float)wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        output[i] = (input[i] > threshold) ? 255 : 0;
    }
    return (unsigned char)threshold;
}

/*---------------------------------------------------------------------------
  3. Sobel Edge Detection on Grayscale Images
  Computes the gradient magnitude from the horizontal (Gx) and vertical (Gy)
  Sobel operators. Borders are not processed.
---------------------------------------------------------------------------*/
void sobel_edge_seq(unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = -input[(y-1)*width + (x-1)] - 2 * input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
                     + input[(y-1)*width + (x+1)] + 2 * input[y*width + (x+1)] + input[(y+1)*width + (x+1)];
            int gy = -input[(y-1)*width + (x-1)] - 2 * input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                     + input[(y+1)*width + (x-1)] + 2 * input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
            int magnitude = (int)sqrt((double)(gx * gx + gy * gy));
            if (magnitude > 255)
                magnitude = 255;
            output[y * width + x] = (unsigned char)magnitude;
        }
    }
}

void sobel_edge_par(unsigned char* input, unsigned char* output, int width, int height) {
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = -input[(y-1)*width + (x-1)] - 2 * input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
                     + input[(y-1)*width + (x+1)] + 2 * input[y*width + (x+1)] + input[(y+1)*width + (x+1)];
            int gy = -input[(y-1)*width + (x-1)] - 2 * input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                     + input[(y+1)*width + (x-1)] + 2 * input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
            int magnitude = (int)sqrt((double)(gx * gx + gy * gy));
            if (magnitude > 255)
                magnitude = 255;
            output[y * width + x] = (unsigned char)magnitude;
        }
    }
}

/*---------------------------------------------------------------------------
  Main Function
---------------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s input_image.(png|jpg|jpeg)\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    // Load the image (preserving original channels)
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Error loading image %s\n", argv[1]);
        return 1;
    }
    printf("Loaded image: %dx%d, %d channels\n\n", width, height, channels);

    // Display menu for task selection
    int option;
    printf("Select an image processing task:\n");
    printf("  1. Gaussian Blur (Color)\n");
    printf("  2. Otsu Thresholding (Grayscale)\n");
    printf("  3. Sobel Edge Detection (Grayscale)\n");
    printf("  4. All tasks\n");
    printf("Enter your choice: ");
    scanf("%d", &option);
    printf("\n");

    double t_start, t_end;

    // Option 1 or 4: Gaussian Blur (Color)
    if (option == 1 || option == 4) {
        unsigned char *gauss_seq = (unsigned char*)calloc(width * height * channels, sizeof(unsigned char));
        unsigned char *gauss_par = (unsigned char*)calloc(width * height * channels, sizeof(unsigned char));
        if (!gauss_seq || !gauss_par) {
            fprintf(stderr, "Memory allocation error for Gaussian outputs.\n");
            stbi_image_free(image);
            return 1;
        }

        // Measure Sequential Gaussian Blur
        t_start = omp_get_wtime();
        gaussian_blur_color_seq(image, gauss_seq, width, height, channels);
        t_end = omp_get_wtime();
        double time_gauss_seq = t_end - t_start;
        printf("Gaussian Blur (Sequential): %f seconds\n", time_gauss_seq);

        // Measure Parallel Gaussian Blur
        t_start = omp_get_wtime();
        gaussian_blur_color_par(image, gauss_par, width, height, channels);
        t_end = omp_get_wtime();
        double time_gauss_par = t_end - t_start;
        printf("Gaussian Blur (Parallel):   %f seconds\n\n", time_gauss_par);

        // Write output images
        stbi_write_png("gaussian_seq.png", width, height, channels, gauss_seq, width * channels);
        stbi_write_png("gaussian_par.png", width, height, channels, gauss_par, width * channels);

        free(gauss_seq);
        free(gauss_par);
    }

    // For Otsu and Sobel, we need a grayscale version of the image.
    // We compute it once.
    unsigned char *gray = (unsigned char*)calloc(width * height, sizeof(unsigned char));
    if (!gray) {
        fprintf(stderr, "Memory allocation error for grayscale conversion.\n");
        stbi_image_free(image);
        return 1;
    }
    rgb_to_grayscale(image, gray, width, height, channels);

    // Option 2 or 4: Otsu Thresholding (Grayscale)
    if (option == 2 || option == 4) {
        unsigned char *otsu_seq = (unsigned char*)calloc(width * height, sizeof(unsigned char));
        unsigned char *otsu_par = (unsigned char*)calloc(width * height, sizeof(unsigned char));
        if (!otsu_seq || !otsu_par) {
            fprintf(stderr, "Memory allocation error for Otsu outputs.\n");
            free(gray);
            stbi_image_free(image);
            return 1;
        }

        // Measure Sequential Otsu Thresholding
        t_start = omp_get_wtime();
        unsigned char thresh_seq = otsu_thresholding_seq(gray, otsu_seq, width, height);
        t_end = omp_get_wtime();
        double time_otsu_seq = t_end - t_start;
        printf("Otsu Thresholding (Sequential): %f seconds, Threshold = %d\n", time_otsu_seq, thresh_seq);

        // Measure Parallel Otsu Thresholding
        t_start = omp_get_wtime();
        unsigned char thresh_par = otsu_thresholding_par(gray, otsu_par, width, height);
        t_end = omp_get_wtime();
        double time_otsu_par = t_end - t_start;
        printf("Otsu Thresholding (Parallel):   %f seconds, Threshold = %d\n\n", time_otsu_par, thresh_par);

        // Write output images (1-channel PNG)
        stbi_write_png("otsu_seq.png", width, height, 1, otsu_seq, width);
        stbi_write_png("otsu_par.png", width, height, 1, otsu_par, width);

        free(otsu_seq);
        free(otsu_par);
    }

    // Option 3 or 4: Sobel Edge Detection (Grayscale)
    if (option == 3 || option == 4) {
        unsigned char *sobel_seq = (unsigned char*)calloc(width * height, sizeof(unsigned char));
        unsigned char *sobel_par = (unsigned char*)calloc(width * height, sizeof(unsigned char));
        if (!sobel_seq || !sobel_par) {
            fprintf(stderr, "Memory allocation error for Sobel outputs.\n");
            free(gray);
            stbi_image_free(image);
            return 1;
        }

        // Measure Sequential Sobel Edge Detection
        t_start = omp_get_wtime();
        sobel_edge_seq(gray, sobel_seq, width, height);
        t_end = omp_get_wtime();
        double time_sobel_seq = t_end - t_start;
        printf("Sobel Edge Detection (Sequential): %f seconds\n", time_sobel_seq);

        // Measure Parallel Sobel Edge Detection
        t_start = omp_get_wtime();
        sobel_edge_par(gray, sobel_par, width, height);
        t_end = omp_get_wtime();
        double time_sobel_par = t_end - t_start;
        printf("Sobel Edge Detection (Parallel):   %f seconds\n\n", time_sobel_par);

        // Write output images (1-channel PNG)
        stbi_write_png("sobel_seq.png", width, height, 1, sobel_seq, width);
        stbi_write_png("sobel_par.png", width, height, 1, sobel_par, width);

        free(sobel_seq);
        free(sobel_par);
    }

    free(gray);
    stbi_image_free(image);

    printf("Processing complete. Check the output images.\n");
    return 0;
}
