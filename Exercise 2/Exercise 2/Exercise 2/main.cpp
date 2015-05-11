// Excersize 1.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// function prototype
void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray);
void computeCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int windowSize, int maxDisp);
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor);

int main(int argc, char* argv[])
{
	cv::Mat img_left = cv::imread("tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_right = cv::imread("tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
	cv::imshow("Tsukuba Left", img_left);
	cv::imshow("Tsukuba Right", img_right);

	// init grayscale mat with 0's
	cv::Mat imgGray_l(img_left.rows, img_left.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat imgGray_r(img_right.rows, img_right.cols, CV_8UC3, cv::Scalar(0,0,0));

	cv::Mat disp_left(img_left.rows, img_left.cols, CV_32FC1, cv::Scalar(0,0,0));
	cv::Mat disp_right(img_right.rows, img_right.cols, CV_32FC1, cv::Scalar(0,0,0));
	
	convertToGrayscale(img_left, imgGray_l);
	convertToGrayscale(img_right, imgGray_r);

	int maxDisp = 15;
	int windowSize = 5;
	int scaleFactor = 16;

	std::vector<cv::Mat> *costVolumeLeft = new std::vector<cv::Mat>(maxDisp);
	std::vector<cv::Mat> *costVolumeRight = new std::vector<cv::Mat>(maxDisp);

	computeCostVolume(imgGray_l, imgGray_r, *costVolumeLeft, *costVolumeRight, windowSize, maxDisp);
	selectDisparity(disp_left, disp_right, *costVolumeLeft, *costVolumeRight, scaleFactor);

	
	//cv::imshow("Gray Left", imgGray_l);
	//cv::imshow("Gray Right", imgGray_r);

	cv::waitKey(0);

	return 0;
}

void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray)
{
	for(int i = 0; i < imgGray.rows; i++)
	{
		for(int j = 0; j < imgGray.cols; j++)
		{
			// L = 0.21*R + 0.72*G + 0.07*B;
			uchar L = 0.21*img.at<cv::Vec3b>(i,j)[2] + 0.72*img.at<cv::Vec3b>(i,j)[1] + 0.07*img.at<cv::Vec3b>(i,j)[0];
			cv::Vec3b gray(L,L,L);
			imgGray.at<cv::Vec3b>(i,j) = gray;
		}
	}
}

// Start with value 15 for maxDisp
// Mat type in costVolume vectors CV_32FC1
// Number of elements in the vector for maxDisp = 15 is 16 (0-15)
void computeCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int windowSize, int maxDisp)
{
	int start = windowSize/2;
	
	for(int i = start; i < imgLeft.rows - start; i++)
	{
		for(int j = start; j < imgLeft.cols - start; j++)
		{
			
		}
	}
}


// maxDisp * scaleDispFactor must be a value below 256
// scaleDispFactor of 16 for a maxDisp value of 15
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor)
{
}

