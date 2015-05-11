// Excersize 1.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "util.h"

// function prototype
void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray);

int main(int argc, char* argv[])
{
	cv::Mat img_left = cv::imread("tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img_right = cv::imread("tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
	cv::imshow("Tsukuba Left", img_left);
	cv::imshow("Tsukuba Right", img_right);

	

	// init grayscale mat with 0's
	cv::Mat imgGray_l(img_left.rows, img_left.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat imgGray_r(img_right.rows, img_right.cols, CV_8UC3, cv::Scalar(0,0,0));

	
	convertToGrayscale(img_left, imgGray_l);
	convertToGrayscale(img_right, imgGray_r);
	
	cv::imshow("Gray Left", imgGray_l);
	cv::imshow("Gray Right", imgGray_r);

	cv::waitKey(0);

	return 0;
}

void convertToGrayscale(const cv::Mat &img, cv::Mat &imgGray)
{
	for(int i = 1; i < imgGray.rows; i++)
	{
		for(int j = 1; j < imgGray.cols; j++)
		{
			// L = 0.21*R + 0.72*G + 0.07*B;
			uchar L = 0.21*img.at<cv::Vec3b>(i,j)[2] + 0.72*img.at<cv::Vec3b>(i,j)[1] + 0.07*img.at<cv::Vec3b>(i,j)[0];
			cv::Vec3b gray(L,L,L);
			imgGray.at<cv::Vec3b>(i,j) = gray;
			
		}
	}
	
}
