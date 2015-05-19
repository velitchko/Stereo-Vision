// Excersize 1.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>


// function prototype
void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray);
void computeCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int windowSize, int maxDisp);
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor);
void show(const std::vector<cv::Mat> &costVolume, const std::string left_right, int maxDisp);

int main(int argc, char* argv[])
{
	cv::Mat img_left = cv::imread("tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_right = cv::imread("tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
	cv::imshow("Tsukuba Left", img_left);
	cv::imshow("Tsukuba Right", img_right);

	// init grayscale mat with 0's
	cv::Mat imgGray_l(img_left.rows, img_left.cols, CV_8UC1, cv::Scalar(0));
	cv::Mat imgGray_r(img_right.rows, img_right.cols, CV_8UC1, cv::Scalar(0));

	cv::Mat disp_left(img_left.rows, img_left.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat disp_right(img_right.rows, img_right.cols, CV_32FC1, cv::Scalar(0));
	
	convertToGrayscale(img_left, imgGray_l);
	convertToGrayscale(img_right, imgGray_r);

	int maxDisp = 15;
	int windowSize = 3;
	int scaleFactor = 16;

	std::vector<cv::Mat> *costVolumeLeft = new std::vector<cv::Mat>(maxDisp);
	std::vector<cv::Mat> *costVolumeRight = new std::vector<cv::Mat>(maxDisp);

	computeCostVolume(imgGray_l, imgGray_r, *costVolumeLeft, *costVolumeRight, windowSize, maxDisp);
	show(*costVolumeLeft, "Left", maxDisp);
	
	selectDisparity(disp_left, disp_right, *costVolumeLeft, *costVolumeRight, scaleFactor);

	
	//cv::imshow("Gray Left", imgGray_l);
	//cv::imshow("Gray Right", imgGray_r);

	cv::waitKey(0);

	return 0;
}

void show(const std::vector<cv::Mat> &costVolume, const std::string left_right, int maxDisp)
{

	for(int i = 0; i < maxDisp; i++)
	{
		std::string str = "Cost Volume " + left_right + " " + std::to_string(i);
		cv::imshow(str, costVolume.at(i));
	}
}

void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray)
{
	for(int i = 0; i < imgGray.rows; i++)
	{
		for(int j = 0; j < imgGray.cols; j++)
		{
			// L = 0.21*R + 0.72*G + 0.07*B;
			uchar L = 0.21*img.at<cv::Vec3b>(i,j)[2] + 0.72*img.at<cv::Vec3b>(i,j)[1] + 0.07*img.at<cv::Vec3b>(i,j)[0];
			imgGray.at<uchar>(i,j) = L;
		}
	}
}

// Start with value 15 for maxDisp
// Mat type in costVolume vectors CV_32FC1
// Number of elements in the vector for maxDisp = 15 is 16 (0-15)
void computeCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int windowSize, int maxDisp)
{

	std::cout << "Left Image Size: " << imgLeft.cols << "," << imgLeft.rows << std::endl;
	std::cout << "Right Image Size: " << imgRight.cols << "," << imgRight.rows << std::endl;
	std::cout << "Max Disparity: " << maxDisp << std::endl;
	std::cout << "Window Size: " << windowSize << std::endl;
	std::cout << "Start at: " << windowSize/2 << std::endl;

	int start = windowSize/2;
	int sad_left;
	int sad_right;

	// iterate through all possible disparities
	for(int d = 0; d < maxDisp; d++)
	{
		costVolumeLeft.at(d) = cv::Mat::zeros(imgLeft.rows, imgLeft.cols, CV_32FC1);
		costVolumeRight.at(d) = cv::Mat::zeros(imgRight.rows, imgRight.cols, CV_32FC1);
		// iterate through image rows (Y)
		for(int i = start; i < imgLeft.rows - start; i++)
		{
			// iterate through image Columns (X)
			for(int j = start; j < imgLeft.cols - start; j++)
			{
				int startY = i - start;
				int startX = j - start;
				int endY = i + start + 1;
				int endX = j + start + 1;	

				sad_left = 0;
				sad_right = 0;
			
				// imgVector(y,x)
				// Compute Sum of Absolute Differences
				for(int k = startY; k <= endY; k++)
				{
					const uchar* left_pixel = imgLeft.ptr<uchar>(k);
					const uchar* right_pixel = imgRight.ptr<uchar>(k);
					
					for(int l = startX; l < endX; l++)
					{
						// Left -> Right
						int gray_left = left_pixel[l];
						int gray_right = right_pixel[l-d];
						// Right -> Left
						int gray_left_r = left_pixel[l-d];
						int gray_right_r = right_pixel[l];
						// SAD
						sad_left += abs(gray_left - gray_right);
						sad_right += abs(gray_right_r - gray_left_r);
					}
				}
				float sad = (float) sad_left / (255.0f*windowSize*windowSize);
				if(sad > 1.0)
				{
					std::cout << "SAD: " << sad << std::endl;
				}
				costVolumeLeft.at(d).at<float>(i,j) = (float) sad_left / 255.0f;
				costVolumeRight.at(d).at<float>(i,j) = (float) sad_right / 255.0f;
			}
		}
	}
}

// maxDisp * scaleDispFactor must be a value below 256
// scaleDispFactor of 16 for a maxDisp value of 15
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor)
{
}

