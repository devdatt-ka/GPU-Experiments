#include "nppfilter.h"

int main(int argc, char* argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image_path> <output_image_path> <filter_type>\n", argv[0]);
        return -1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];
    const char* filterType = argv[3];

    // Load image using OpenCV
    cv::Mat inputImage = cv::imread(inputFileName, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        fprintf(stderr, "Failed to load input image\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int step = inputImage.step;

    NppiSize imageSize;
    imageSize.width = width;
    imageSize.height = height;

    Npp8u* pHostBuffer = inputImage.data;

    Npp8u* pDeviceBuffer = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&pDeviceBuffer, step * height));

    // Copy image data to device
    CHECK_CUDA(cudaMemcpy(pDeviceBuffer, pHostBuffer, step * height, cudaMemcpyHostToDevice));

    // Allocate output image
    Npp8u* d_outputImage = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_outputImage, step * height));

    // Define box filter kernel size
    NppiSize maskSize = { 50, 50 }; // Default mask size
    NppiPoint anchor = { maskSize.width / 2, maskSize.height / 2 }; // Default anchor

    // Apply filter based on the filter type argument
    if (strcmp(filterType, "box") == 0) {
        CHECK_NPP(nppiFilterBox_8u_C3R(pDeviceBuffer, step, d_outputImage, step, imageSize, maskSize, anchor));
    } else if (strcmp(filterType, "gaussian") == 0) {
        // Example: Apply Gaussian filter (you need to implement this)
        NppiMaskSize maskSizeGauss = NPP_MASK_SIZE_5_X_5; // { 50, 50 }; // Gaussian filter mask size
        CHECK_NPP(nppiFilterGauss_8u_C3R(pDeviceBuffer, step, d_outputImage, step, imageSize, maskSizeGauss));
    }
    else if (strcmp(filterType, "sharpen") == 0) {
        // Example: Apply Sharpen filter (you need to implement this)
        CHECK_NPP(nppiFilterSharpen_8u_C3R(pDeviceBuffer, step, d_outputImage, step, imageSize));
    } else {
        fprintf(stderr, "Unknown filter type: %s\n", filterType);
        return -1;
    }

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(pHostBuffer, d_outputImage, step * height, cudaMemcpyDeviceToHost));

    // Save output image using OpenCV
    cv::Mat outputImage(height, width, CV_8UC3);
    memcpy(outputImage.data, pHostBuffer, step * height);
    if (!cv::imwrite(outputFileName, outputImage)) {
        fprintf(stderr, "Failed to save output image\n");
        return -1;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(pDeviceBuffer));
    CHECK_CUDA(cudaFree(d_outputImage));

    // Reset the device and exit
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
