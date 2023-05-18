/***********************************************************
* Based on "Noise Aware Image Assessment metric based Auto Exposure Control" by "Uk Cheol Shin, KAIST RCV LAB"
***********************************************************/
#pragma once

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdarg.h>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#define TOL 1e-30 /* smallest value allowed in cholesky_decomp() */

class ImageQualityEstimator
{
public :

	ImageQualityEstimator();
	//double GetImgQualityValue(const cv::Mat &image, float resize_factor = 1.0, int flag_boost = 0);

	//@brief Calculate Image gradient based on Sobel operator
	double CalImgGradient(const cv::Mat &grayImg, cv::Mat &output_gradient);
	//@brief Calculate Spatial distribution Gradient
	double CalBalanceGradient(const cv::Mat &gradImg, int lambda = 10*10*10, float delta = 0.6, int gridnum = 10);
	//@brief Calculate Entropy
	float CalImageEntropy(const cv::Mat src);
	//@brief Calculate Noise Value
	double CalImageNoiseVariance(const cv::Mat& rgbImg, int flag_boost = 0);
	//@brief Calculate Sharpess
	double CalSharpness(const cv::Mat& grayImg);

private:
	float alpha;
	float beta;
	double CurIntensity;
	double CurGradInfo;
	double CurEntroInfo;
	double CurNoiseInfo;
};
