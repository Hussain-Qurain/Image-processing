#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

// Apply a simple Gaussian blur manually (for demonstration)
// In practice, you can start with cv::GaussianBlur, then replace it with a custom kernel for parallelization.
void gaussianBlurCPU(const cv::Mat &input, cv::Mat &output)
{
    // Simple 3x3 Gaussian kernel
    float kernel[3][3] = {
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int px = x + kx;
                    int py = y + ky;
                    sum += input.at<uchar>(py, px) * kernel[ky + 1][kx + 1];
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

int main()
{
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

    // benchmark processing 100 frames for example
    int num_frames = 100;
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
        gaussianBlurCPU(gray, blurImg);

        // Sobel edge detection (CPU)
        edgeImg.setTo(0);
        sobelEdgeCPU(blurImg, edgeImg);

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
