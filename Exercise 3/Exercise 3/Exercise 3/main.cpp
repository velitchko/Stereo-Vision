#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "guidedfilter.h"
#include <iostream>
#include <string>
#include <Windows.h>

using namespace cv;
using namespace std;

// function prototype
void convertToGrayscale(const Mat &img, Mat &imgGray);
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int windowSize, int maxDisp);
void selectDisparity(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int scaleDispFactor);
void aggregateCostVolume(const Mat &imgLeft, const Mat &imgRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int r, double eps);
void refineDisparity(Mat &dispLeft, Mat &dispRight, int scaleFactor);
void show(const vector<Mat> &costVolume, const string left_right, int maxDisp);

int main(int argc, char* argv[])
{
	bool running = true;
	
	while (running && GetAsyncKeyState(VK_ESCAPE) == 0) {
		cout << "choose image (1=tsukuba, 2=cones, 3=teddy, 4=venus)" << endl;

		string names[] = { "tsukuba", "cones", "teddy", "venus" };

		Mat img_left, img_right;
		string line; istringstream str;
		unsigned int img, maxDisp, windowSize;
		getline(cin, line); str = istringstream(line); str >> img; img--;
		if (!str || img > names->size() || img < 0) { img = 0; }

		
		cout << "Window size? (default = 5)" << endl;
		getline(cin, line); str = istringstream(line); str >> windowSize;
		if (windowSize % 2 == 0) { windowSize++; }
		if (!str || windowSize < 0) { windowSize = 5; }
		
		cout << "Maximum Disparity? (default = 15)" << endl;
		getline(cin, line); str = istringstream(line); str >> maxDisp;
		if (!str || maxDisp < 0) { maxDisp = 15; }
		maxDisp++;

		cout << "Image chosen: " << names[img] << endl;
		cout << "Window size: " << windowSize << endl;
		cout << "Max Disp: " << maxDisp-1 << endl;
		
		img_left = imread(names[img] + "_left.png", CV_LOAD_IMAGE_COLOR);
		img_right = imread(names[img] + "_right.png", CV_LOAD_IMAGE_COLOR);

		imshow("Image Left", img_left);
		imshow("Image Right", img_right);
		waitKey(1);
		

		// init grayscale mat with 0's
		Mat imgGray_l = Mat::zeros(img_left.rows, img_left.cols, CV_8UC1);
		Mat imgGray_r = Mat::zeros(img_right.rows, img_right.cols, CV_8UC1);

		Mat disp_left = Mat::zeros(img_left.rows, img_left.cols, CV_8UC1);
		Mat disp_right = Mat::zeros(img_right.rows, img_right.cols, CV_8UC1);

		convertToGrayscale(img_left, imgGray_l);
		convertToGrayscale(img_right, imgGray_r);

		int scaleFactor = 255 / maxDisp;

		vector<Mat> *costVolumeLeft = new vector<Mat>(maxDisp);
		vector<Mat> *costVolumeRight = new vector<Mat>(maxDisp);

		double t = (double)getTickCount();
		computeCostVolume(imgGray_l, imgGray_r, *costVolumeLeft, *costVolumeRight, windowSize, maxDisp);
		t = ((double)getTickCount() - t) / getTickFrequency();


		aggregateCostVolume(img_left, img_right, *costVolumeLeft, *costVolumeRight, 9, 0.01*0.01); //TODO parameter ausprobieren
		
		selectDisparity(disp_left, disp_right, *costVolumeLeft, *costVolumeRight, scaleFactor);
		
		refineDisparity(disp_left, disp_right, scaleFactor);


		
		imshow("Disparity Left", disp_left);
		imshow("Disparity Right", disp_right);

		waitKey(1);

		cout << "Done in " << t << "seconds. Save files? (y/n)" << endl;
		getline(cin, line);
		if (line == "y") {
			imwrite(names[img] + "_disp_left.png", disp_left);
			imwrite(names[img] + "_disp_right.png", disp_right);
			cout << "Done!" << endl;
		}

		cout << "restart? (y/n)" << endl;
		getline(cin, line); running = line == "y";

	}
	return 0;
}

void show(const vector<Mat> &costVolume, const string left_right, int maxDisp)
{
	for(int i = 0; i < maxDisp; i++)
	{
		string str = "Cost Volume " + left_right + " " + to_string(i);
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
			uchar L = (uchar)(0.21*img.at<Vec3b>(i,j)[2] + 0.72*img.at<Vec3b>(i,j)[1] + 0.07*img.at<Vec3b>(i,j)[0]);
			imgGray.at<uchar>(i,j) = L;
		}
	}
}

// Start with value 15 for maxDisp
// Mat type in costVolume vectors CV_32FC1
// Number of elements in the vector for maxDisp = 15 is 16 (0-15)
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int windowSize, int maxDisp)
{

	cout << "Left Image Size: " << imgLeft.cols << "," << imgLeft.rows << endl;
	cout << "Right Image Size: " << imgRight.cols << "," << imgRight.rows << endl;
	cout << "Max Disparity: " << maxDisp << endl;
	cout << "Window Size: " << windowSize << endl;
	cout << "Start at: " << windowSize/2 << endl;
	int start = windowSize/2;
	int sad_left, sad_right;
	float normalize = 1.0f / (255 * windowSize*windowSize);

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
				if (j == start) {
					sad_left = 0;
					sad_right = 0;

					// Compute Sum of Absolute Differences
					for (int l = -start; l <= start; l++) {
						for (int k = -start; k <= start; k++) {
							int x = j + l,
								y = i + k,
								diffLeft = d,
								diffRight = d;

							if (x - d > 0) {
								diffLeft = abs(imgLeft.at<uchar>(y, x) - imgRight.at<uchar>(y, x - d));
							}
							if (x + d < imgLeft.cols) {
								diffRight = abs(imgLeft.at<uchar>(y, x + d) - imgRight.at<uchar>(y, x));
							}
							sad_left += diffLeft;
							sad_right += diffRight;
						}
					}
				} else {
					//SUBTRACT LEFT COLUMN
					for (int k = -start; k <= start; k++) {
						int x = j - start - 1,
							y = i + k,
							diffLeft = d,
							diffRight = d;

						if (x - d > 0) {
							diffLeft = abs(imgLeft.at<uchar>(y, x) - imgRight.at<uchar>(y, x - d));
						}
						if (x + d < imgLeft.cols) {
							diffRight = abs(imgLeft.at<uchar>(y, x + d) - imgRight.at<uchar>(y, x));
						}
						sad_left -= diffLeft;
						sad_right -= diffRight;
					}

					//ADD RIGHT COLUMN
					for (int k = -start; k <= start; k++) {
						int x = j + start,
							y = i + k,
							diffLeft = d,
							diffRight = d;

						if (x - d > 0) {
							diffLeft = abs(imgLeft.at<uchar>(y, x) - imgRight.at<uchar>(y, x - d));
						}
						if (x + d < imgLeft.cols) {
							diffRight = abs(imgLeft.at<uchar>(y, x + d) - imgRight.at<uchar>(y, x));
						}
						sad_left += diffLeft;
						sad_right += diffRight;
					}
				}
				// save to costVolume
				costVolumeLeft.at(d).at<float>(i, j) = normalize * sad_left;
				costVolumeRight.at(d).at<float>(i, j) = normalize * sad_right;
			}
		}
	}
}


// maxDisp * scaleDispFactor must be a value below 256
// scaleDispFactor of 16 for a maxDisp value of 15
void selectDisparity(Mat &dispLeft, Mat &dispRight,	vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int scaleDispFactor)
{
	Mat minCostLeft = Mat::ones(dispLeft.rows, dispLeft.cols, CV_32FC1);
	Mat minCostRight = Mat::ones(dispLeft.rows, dispLeft.cols, CV_32FC1);

	for (unsigned int d = 0; d < costVolumeLeft.size(); d++) {
		Mat &layerLeft = costVolumeLeft.at(d);
		Mat &layerRight = costVolumeRight.at(d);

		for (int x = 0; x < layerLeft.cols; x++) {
			for (int y = 0; y < layerLeft.rows; y++) {
				float valueLeft = layerLeft.at<float>(y, x);
				float valueRight = layerRight.at<float>(y, x);

				if (valueLeft < minCostLeft.at<float>(y, x)) {
					minCostLeft.at<float>(y, x) = valueLeft;
					dispLeft.at<uchar>(y, x) = min(254,d * scaleDispFactor); //255 reserved as invalid
				}
				if (valueRight < minCostRight.at<float>(y, x)) {
					minCostRight.at<float>(y, x) = valueRight;
					dispRight.at<uchar>(y, x) = min(254,d * scaleDispFactor); //255 reserved as invalid
				}
			}
		}
	}
}

//Window size of guided filter = 2*r+1
//r = 9
//eps = 0.01^2
void aggregateCostVolume(const Mat &imgLeft, const Mat &imgRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int r, double eps)
{
	int disparities = costVolumeLeft.size();
	eps *= 255 * 255;

	for(int i = 0; i < disparities; i++)
	{
		Mat &p_l = costVolumeLeft.at(i);
		Mat &p_r = costVolumeRight.at(i);
		
		p_l = guidedFilter(imgLeft, p_l, r, eps);
		p_r = guidedFilter(imgRight, p_r, r, eps);
		
	}

}

struct Coords {
	int y;
	int x;
};

void refineDisparity(Mat &dispLeft, Mat &dispRight, int scaleFactor)
{
	int epsilon = 1;
	int invalid = 255;

	vector<Coords> left_invalids, right_invalids;

	uchar left, right, leftD, rightD;
	for (int y = 0; y < dispLeft.rows; y++) {
		for (int x = 0; x < dispLeft.cols; x++) {
			left = dispLeft.at<uchar>(y, x) / scaleFactor;
			right = dispRight.at<uchar>(y, x) / scaleFactor;
			leftD = dispLeft.at<uchar>(y, x + right) / scaleFactor;
			rightD = dispRight.at<uchar>(y, x - left) / scaleFactor;

			if (abs(left - rightD) > epsilon) {
				dispLeft.at<uchar>(y, x) = invalid;
				left_invalids.push_back({ y, x });
			}

			if (abs(leftD - right) > epsilon) {
				dispRight.at<uchar>(y, x) = invalid;
				right_invalids.push_back({ y, x });
			}
		}
	}

	for (int i = 0; i < 2; i++) {
		vector<Coords>& invalids = i == 0 ? left_invalids : right_invalids;
		Mat& disp = i == 0 ? dispLeft : dispRight;
		
		for (auto coords : invalids) {
			uchar dl, dr;
			int currentX = coords.x;
			uchar currentDisp;

			while (currentX >= 0) {
				currentDisp = disp.at<uchar>(coords.y, currentX);
				if (currentDisp != invalid) {
					dl = currentDisp;
					break;
				}
				currentX--;
			}

			currentX = coords.x;
			while (currentX < disp.cols) {
				currentDisp = disp.at<uchar>(coords.y, currentX);
				if (currentDisp != invalid) {
					dr = currentDisp;
					break;
				}
				currentX++;
			}

			disp.at<uchar>(coords.y, coords.x) = min(dl, dr);
		}
		
		Mat src = disp.clone();
		medianBlur(src, disp, 3); //TODO
	}

	
}