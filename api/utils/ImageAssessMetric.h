/***********************************************************
* @PROJECT : Noise Aware Image Assessment metric based Auto Exposure Control
* @TITLE   : Image Assessment Metric 
* @AUTHOR  : Uk Cheol Shin, KAIST RCV LAB
* @DATE    : 2018-08-14
* @BRIEF   : Calculate Entropy weighted gradient value
* @UPDATE  : 2019-01-23
* @BRIEF   : Update Control Part to use Nelder-Mead Algorithm.
***********************************************************/
#ifndef __IMAGE_QUALITY_ESTIMATOR_H__
#define __IMAGE_QUALITY_ESTIMATOR_H__

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

#define TIME_DEBUG 0
#define PLOT_DEBUG 0
// #define DEBUG 0

#define TOL 1e-30 /* smallest value allowed in cholesky_decomp() */

class IMAGE_QUALITY_ESTIMATOR
{
private :

	//@brief Calculate Image gradient based on Sobel operator
	double CalImgGradient(const cv::Mat &grayImg, cv::Mat &output_gradient);
	//@brief Calculate Spatial distribution Gradient
	double CalBalanceGradient(const cv::Mat &gradImg, int lambda = 10*10*10, float delta = 0.6, int gridnum = 10);
	//@brief Calculate Entropy
	float CalImageEntropy(const cv::Mat src);
	//@brief Calculate Noise Value
	double CalImageNoiseVariance(const cv::Mat& rgbImg, int flag_boost = 0);

	//@brief for Debug
	void display_img(cv::Mat &image);
	//@brief for Debug
	void print_img_info(cv::Mat &image);

public :
	float alpha;
	float beta;

	IMAGE_QUALITY_ESTIMATOR();
	double CurIntensity;
	double CurGradInfo;
	double CurEntroInfo;
	double CurNoiseInfo;
	double GetImgQualityValue(const cv::Mat &image, float resize_factor = 1.0,int flag_boost = 0);
};



#endif // define __IMAGE_QUALITY_ESTIMATOR_H__
