#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>


// Apply a simple Gaussian blur manually (for demonstration)
// In practice, you can start with cv::GaussianBlur, then replace it with a custom kernel for parallelization.
void gaussianBlurCPU(const cv::Mat &input, cv::Mat &output)
{
    // 5x5 Gaussian kernel
    float kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}};

    // Normalize the kernel
    float sumKernel = 0.0f;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            sumKernel += kernel[i][j];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            kernel[i][j] /= sumKernel;

    // Apply kernel to the image
    for (int y = 2; y < input.rows - 2; y++) // Adjusted for 5x5 kernel
    {
        for (int x = 2; x < input.cols - 2; x++) // Adjusted for 5x5 kernel
        {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ky++) // Adjusted for 5x5 kernel
            {
                for (int kx = -2; kx <= 2; kx++) // Adjusted for 5x5 kernel
                {
                    int px = x + kx;
                    int py = y + ky;
                    sum += input.at<uchar>(py, px) * kernel[ky + 2][kx + 2];
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }
}

// Apply Sobel edge detection manually on CPU
void sobelEdgeCPU(const cv::Mat &input, cv::Mat &output)
{
    // Sobel kernels for X and Y directions
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    int gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            int sumX = 0;
            int sumY = 0;
            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int px = x + kx;
                    int py = y + ky;
                    uchar val = input.at<uchar>(py, px);
                    sumX += val * gx[ky + 1][kx + 1];
                    sumY += val * gy[ky + 1][kx + 1];
                }
            }
            int magnitude = std::min(255, std::max(0, (int)std::sqrt(sumX * sumX + sumY * sumY)));
            output.at<uchar>(y, x) = (uchar)magnitude;
        }
    }
}

// Optimized Gaussian blur using OpenACC
void gaussianBlurGPU(const cv::Mat &input, cv::Mat &output) {
    // 5x5 Gaussian kernel
    float kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}};

    // Normalize kernel (done on CPU as it's small)
    float sumKernel = 0.0f;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            sumKernel += kernel[i][j];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            kernel[i][j] /= sumKernel;

    const int rows = input.rows;
    const int cols = input.cols;
    const uchar* in_ptr = input.ptr<uchar>(0);
    uchar* out_ptr = output.ptr<uchar>(0);

    #pragma acc data copyin(in_ptr[0:rows*cols], kernel[0:5][0:5]) copyout(out_ptr[0:rows*cols])
    {
        #pragma acc parallel loop collapse(2) present(in_ptr, out_ptr, kernel)
        for (int y = 2; y < rows - 2; y++) {
            for (int x = 2; x < cols - 2; x++) {
                float sum = 0.0f;
                
                #pragma acc loop collapse(2) reduction(+:sum)
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        sum += in_ptr[py * cols + px] * kernel[ky + 2][kx + 2];
                    }
                }
                out_ptr[y * cols + x] = static_cast<uchar>(sum);
            }
        }
    }
}

// Optimized Sobel edge detection using OpenACC
void sobelEdgeGPU(const cv::Mat &input, cv::Mat &output) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    const int rows = input.rows;
    const int cols = input.cols;
    const uchar* in_ptr = input.ptr<uchar>(0);
    uchar* out_ptr = output.ptr<uchar>(0);

    #pragma acc data copyin(in_ptr[0:rows*cols], gx[0:3][0:3], gy[0:3][0:3]) copyout(out_ptr[0:rows*cols])
    {
        #pragma acc parallel loop collapse(2) present(in_ptr, out_ptr, gx, gy)
        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                int sumX = 0;
                int sumY = 0;

                #pragma acc loop collapse(2) reduction(+:sumX,sumY)
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        uchar val = in_ptr[py * cols + px];
                        sumX += val * gx[ky + 1][kx + 1];
                        sumY += val * gy[ky + 1][kx + 1];
                    }
                }
                int magnitude = std::min(255, std::max(0, (int)std::sqrt(sumX * sumX + sumY * sumY)));
                out_ptr[y * cols + x] = (uchar)magnitude;
            }
        }
    }
}

int main()
{
    acc_init(acc_device_default);

    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened())
    {
        std::cerr << "Error: Cannot open camera.\n";
        return -1;
    }

    // Capture one frame to get dimensions
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
    {
        std::cerr << "Error: Empty frame from camera.\n";
        return -1;
    }

    // Convert to gray
    cv::Mat gray, blurImg, edgeImg;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    blurImg = gray.clone();
    edgeImg = gray.clone();

    // Warm-up run (to allow camera to stabilize, etc.)
    for (int i = 0; i < 10; i++)
    {
        cap >> frame;
    }

    // benchmark processing x frames
    int num_frames = 200;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_frames; i++)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Gaussian blur (CPU)
        blurImg.setTo(0);
        gaussianBlurGPU(gray, blurImg);

        // Sobel edge detection (CPU)
        edgeImg.setTo(0);
        sobelEdgeGPU(blurImg, edgeImg);

        // Optional threshold (CPU)
        cv::threshold(edgeImg, edgeImg, 50, 255, cv::THRESH_BINARY);

        // Display
        cv::imshow("Edge Detection", edgeImg);
        if (cv::waitKey(1) == 27)
            break; // Press ESC to exit
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double fps = num_frames / (elapsedMs / 1000.0);

    std::cout << "Processed " << num_frames << " frames in " << elapsedMs << " ms. Approx FPS: " << fps << "\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
