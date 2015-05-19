// Excersize 1.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;

// function prototype
void convertToGrayscale(const Mat &img, Mat &imgGray);
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight, int windowSize, int maxDisp);
void selectDisparity(Mat &dispLeft, Mat &dispRight, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight, int scaleDispFactor);
void show(const std::vector<Mat> &costVolume, const std::string left_right, int maxDisp);

int main(int argc, char* argv[])
{
	Mat img_left = imread("tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
	Mat img_right = imread("tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
	imshow("Tsukuba Left", img_left);
	imshow("Tsukuba Right", img_right);

	// init grayscale mat with 0's
	Mat imgGray_l = Mat::zeros(img_left.rows, img_left.cols, CV_8UC1);
	Mat imgGray_r = Mat::zeros(img_right.rows, img_right.cols, CV_8UC1);

	Mat disp_left = Mat::zeros(img_left.rows, img_left.cols, CV_32FC1);
	Mat disp_right = Mat::zeros(img_right.rows, img_right.cols, CV_32FC1);
	
	convertToGrayscale(img_left, imgGray_l);
	convertToGrayscale(img_right, imgGray_r);

	int maxDisp = 15;
	int windowSize = 5;
	int scaleFactor = 16;

	std::vector<Mat> *costVolumeLeft = new std::vector<Mat>(maxDisp);
	std::vector<Mat> *costVolumeRight = new std::vector<Mat>(maxDisp);

	computeCostVolume(imgGray_l, imgGray_r, *costVolumeLeft, *costVolumeRight, windowSize, maxDisp);
	show(*costVolumeLeft, "Left", maxDisp);
	
	selectDisparity(disp_left, disp_right, *costVolumeLeft, *costVolumeRight, scaleFactor);

	
	//imshow("Gray Left", imgGray_l);
	//imshow("Gray Right", imgGray_r);

	waitKey(0);

	return 0;
}

void show(const std::vector<Mat> &costVolume, const std::string left_right, int maxDisp)
{

	for(int i = 0; i < maxDisp; i++)
	{
		std::string str = "Cost Volume " + left_right + " " + std::to_string(i);
		imshow(str, costVolume.at(i));
	}
}

void convertToGrayscale(const Mat &img, Mat &imgGray)
{
	for(int i = 0; i < imgGray.rows; i++)
	{
		for(int j = 0; j < imgGray.cols; j++)
		{
			// L = 0.21*R + 0.72*G + 0.07*B;
			uchar L = 0.21*img.at<Vec3b>(i,j)[2] + 0.72*img.at<Vec3b>(i,j)[1] + 0.07*img.at<Vec3b>(i,j)[0];
			imgGray.at<uchar>(i,j) = L;
		}
	}
}

// Start with value 15 for maxDisp
// Mat type in costVolume vectors CV_32FC1
// Number of elements in the vector for maxDisp = 15 is 16 (0-15)
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight, int windowSize, int maxDisp)
{

	std::cout << "Left Image Size: " << imgLeft.cols << "," << imgLeft.rows << std::endl;
	std::cout << "Right Image Size: " << imgRight.cols << "," << imgRight.rows << std::endl;
	std::cout << "Max Disparity: " << maxDisp << std::endl;
	std::cout << "Window Size: " << windowSize << std::endl;
	std::cout << "Start at: " << windowSize/2 << std::endl;
	int start = windowSize/2;
	float sad;

	// iterate through all possible disparities
	for(int d = 0; d < maxDisp; d++)
	{
		costVolumeLeft.at(d) = Mat::zeros(imgLeft.rows, imgLeft.cols, CV_32FC1);
		costVolumeRight.at(d) = Mat::zeros(imgRight.rows, imgRight.cols, CV_32FC1);

		// iterate through image rows (Y)
		for(int i = start; i < imgLeft.rows - start; i++)
		{
			// iterate through image Columns (X)
			for(int j = start; j < imgLeft.cols - start; j++)
			{
				sad = 0;
				//std::cout << "P(" << j << "," << i << ") \t Q(" << j << "," << i-d << ")\t d:" << d << std::endl; //<< imgLeft.at<Vec3b>(j,i)[0] + " << imgRight.at<Vec3b>(j,i-d)[0] 
				
				// imgVector(y,x)
				// Compute Sum of Absolute Differences
				for(int k = -start; k <= start; k++)
				{
					for(int l = -start; l < start; l++)
					{
						sad += abs(imgLeft.at<uchar>(i+k,j+l) - imgRight.at<uchar>(i+k, j+l-d));
					}
				}

				// save to costVolume
				costVolumeLeft.at(d).at<float>(i, j) = sad / (255*windowSize*windowSize);
				//costVolumeRight.at(d).at<Vec3b>(j,i) = Vec3b(sad,sad,sad);
			}
		}
	}
}


// maxDisp * scaleDispFactor must be a value below 256
// scaleDispFactor of 16 for a maxDisp value of 15
void selectDisparity(Mat &dispLeft, Mat &dispRight, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight, int scaleDispFactor)
{
}

