/***********************************************************
 * @PROJECT : Noise Aware Image Assessment metric based Auto Exposure Control
 * @TITLE   : Entropy weighted gradient
 * @AUTHOR  : Uk Cheol Shin, KAIST RCV LAB
 * @DATE    : 2018-08-14
 * @BRIEF   : Calculate Entropy weighted gradient value
 * @UPDATE  : 2019-01-23
 * @BRIEF   : Decouple control part and image quality assessment metric part
 ***********************************************************/
#include "ImageQualityEstimator.hpp"

ImageQualityEstimator::ImageQualityEstimator()
    : alpha(0.5), beta(0.4), CurIntensity(0), CurGradInfo(0), CurEntroInfo(0), CurNoiseInfo(0)
{
}

double ImageQualityEstimator::CalBalanceGradient(const cv::Mat &grayImg, int lambda, float delta, int gridnum)
{
    if (!grayImg.data || grayImg.dims != 2)
    {
        return 0;
    }
    int Img_width = grayImg.rows;
    int Img_height = grayImg.cols;

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // 1) Get the horizontal & vertical gradient
    cv::Sobel(grayImg, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grayImg, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    // 2) Calculate normalized gradient magnitude
    cv::Mat GradientImg(Img_width, Img_height, CV_32F);
    cv::Size size = grad_x.size();
    if (grad_x.isContinuous() && grad_y.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    const float *gx, *gy;
    float *out;
    float max = sqrt(16 * 255 * 255 + 16 * 255 * 255); // max value out of a sobel filter is 4*255
    for (int i = 0; i < size.height; ++i)
    {
        gx = grad_x.ptr<float>(i);
        gy = grad_y.ptr<float>(i);
        out = GradientImg.ptr<float>(i);
        for (int j = 0; j < size.width; ++j)
        {
            out[j] = std::sqrt(gx[j] * gx[j] + gy[j] * gy[j]);
            if (out[j] > max)
            {
                std::cout << "Gradient exceeds the maximum value. Clamped." << std::endl;
                out[j] = max;
            }
        }
    }
    GradientImg /= max;

    // 3) Calculate gradient information
    float N_inv = 1 / (std::log(lambda * (1 - delta) + 1));
    cv::Mat GradientInforImg(Img_width, Img_height, CV_32F);

    size = GradientImg.size();
    if (GradientImg.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    const float *mag;
    for (int i = 0; i < size.height; ++i)
    {
        mag = GradientImg.ptr<float>(i);
        out = GradientInforImg.ptr<float>(i);
        for (int j = 0; j < size.width; ++j)
        {
            if (mag[j] >= delta)
            {
                out[j] = N_inv * std::log(lambda * (mag[j] - delta) + 1);
            }
            else
            {
                out[j] = 0.0;
            }
        }
    }

    // 4) Calculate Gradient Spatial information.
    cv::Mat GradientGridInfor(gridnum, gridnum, CV_32F);
    int Len_width = round(Img_width / gridnum);
    int Len_heigh = round(Img_height / gridnum);
    for (int i = 0; i < gridnum; i++)
    {
        for (int j = 0; j < gridnum; j++)
        {
            cv::Mat imageROI = GradientInforImg(cv::Rect(Len_heigh * i, Len_width * j, Len_heigh, Len_width));
            auto mean = cv::mean(imageROI)[0];
            // if (mean != 0)
            //     std::cout << "mean: " << mean << std::endl;
            GradientGridInfor.at<float>(j, i) = mean;
        }
    }

    cv::Mat mean_grid;
    cv::Mat std_grid;
    cv::meanStdDev(GradientGridInfor, mean_grid, std_grid);
    if (mean_grid.at<double>(0) <= 0.0)
        return 10;
    double meandivstd = mean_grid.at<double>(0) / std_grid.at<double>(0);

    return meandivstd;
}

double ImageQualityEstimator::CalImgGradient(const cv::Mat &grayImg, cv::Mat &output_gradient)
{
    if (!grayImg.data || grayImg.dims != 2)
        return 0;

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // input, output, image_type, kernel, kernel size, scale and offset
    // 1) Get the horizontal & vertical gradientZ
    // cv::GaussianBlur(grayImg, grayImg, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grayImg, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grayImg, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    // calculate gradient magnitude
    cv::Size size = grad_x.size();
    if (grad_x.isContinuous() && grad_y.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    const float *gx, *gy;
    float *out;
    // max value out of a sobel filter is 4*255
    float max = sqrt(16 * 255 * 255 + 16 * 255 * 255);
    for (int i = 0; i < size.height; ++i)
    {
        gx = grad_x.ptr<float>(i);
        gy = grad_y.ptr<float>(i);
        out = output_gradient.ptr<float>(i);
        for (int j = 0; j < size.width; ++j)
        {
            out[j] = std::sqrt(gx[j] * gx[j] + gy[j] * gy[j]);
            if (out[j] > max)
            {
                std::cout << "Gradient exceeds the maximum value. Clamped." << std::endl;
                out[j] = max;
            }
        }
    }
    return 1;
}

double ImageQualityEstimator::CalImageNoiseVariance(const cv::Mat &rgbImg, int flag_boost)
{
    double noise_level[3];
    int channgel_num = rgbImg.channels();
    int Img_width = rgbImg.rows;
    int Img_height = rgbImg.cols;
    float p = 0.10;
    cv::Mat bgr_img[3];
    cv::split(rgbImg, bgr_img);

    for (int i = 0; i < channgel_num; i++)
    {
        if (flag_boost == 1)
            i = 1;
        cv::Mat single_image = bgr_img[i];
        cv::Mat gradientImag(Img_width, Img_height, CV_32F);
        CalImgGradient(single_image, gradientImag);

        // calculate adptive edge threshold
        std::vector<double> Grad_1D = gradientImag.reshape(0, 1);
        cv::sort(Grad_1D, Grad_1D, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
        int Treshold_index = round(Img_width * Img_height * p);
        double Edge_Threshold = Grad_1D.at(Treshold_index);

        // laplacian image
        cv::Mat Noise_mask = (cv::Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
        cv::Mat Laplacian_img(Img_width, Img_height, CV_32F);
        single_image.convertTo(single_image, CV_32F);
        cv::filter2D(single_image, Laplacian_img, -1, Noise_mask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

        cv::Size size = gradientImag.size();
        if (gradientImag.isContinuous() && single_image.isContinuous() && Laplacian_img.isContinuous())
        {
            size.width *= size.height;
            size.height = 1;
        }
        // homogeneous mask
        int Area_num = 0;
        double noise_tmp = 0;
        cv::Mat robust_laplacian_img(Img_width, Img_height, CV_32F);
        for (int i = 0; i < size.height; i++)
        {
            float *grad_ptr = gradientImag.ptr<float>(i);
            float *inten_ptr = single_image.ptr<float>(i);
            float *lapla_ptr = Laplacian_img.ptr<float>(i);
            for (int j = 0; j < size.width; j++)
            {
                if (grad_ptr[j] <= Edge_Threshold)
                {
                    if ((inten_ptr[j] <= 235) && (15 <= inten_ptr[j]))
                    {
                        noise_tmp += abs(lapla_ptr[j]);
                        Area_num++;
                    }
                }
            }
        }

        // noise level estimation
        if (Area_num > 10)
            noise_level[i] = sqrt(M_PI / 2) * 1 / (6 * Area_num) * noise_tmp;
        else
            noise_level[i] = 20;

        if (flag_boost == 1)
            break;
    }
    if (flag_boost == 1)
        return noise_level[1];
    else
        return (noise_level[0] + 2 * noise_level[1] + noise_level[2]) / 4;
}

float ImageQualityEstimator::CalImageEntropy(const cv::Mat src)
{
    cv::Mat hist;
    const int histSize = 256;

    // Compute the histograms:
    float range[] = {0, histSize};
    const float *histRange = {range};

    // images, number of images, channels, mask, hist, dim, histsize, ranges,uniform, accumulate
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // compute entropy
    float entropy_value = 0;
    float total_size = src.rows * src.cols; // total size of all symbols in an image

    float *sym_occur = hist.ptr<float>(0); // the number of times a sybmol has occured
    for (int i = 0; i < histSize; i++)
    {
        if (sym_occur[i] > 0) // log of zero goes to infinity
        {
            entropy_value += (sym_occur[i] / total_size) * (std::log2(total_size / sym_occur[i]));
        }
    }

    entropy_value /= 8.0;

    hist.release();

    return entropy_value;
}

double ImageQualityEstimator::GetImgQualityValue(const cv::Mat &image, float resize_factor /*= 1.0*/, int flag_boost)
{
    static const float norm_8bits_mult = 1.0f / 255.0f;
    cv::Mat resized_image;
    cv::Mat gray_image;
    // 0. resize
    cv::Size size = image.size();
    cv::resize(image, resized_image, cv::Size(resize_factor * size.width, resize_factor * size.height));
    if (resized_image.channels() == 3)
    {
        cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else if (resized_image.channels() == 1)
    {
        gray_image = resized_image;
    }

    // 0. Additional term for the control exception handling
    const double Intensity_Value = cv::mean(gray_image)(0);
    CurIntensity = Intensity_Value * norm_8bits_mult; // normalize to 1

    // 1. Compute gradient domain value
    const double Gradient_value = CalBalanceGradient(gray_image, 10 * 10 * 10, 0.06, 10);
    CurGradInfo = Gradient_value;

    // 2. Compute entropy domain value
    const float Entropy_value = CalImageEntropy(gray_image);
    CurEntroInfo = Entropy_value;

    // 3. Compute noise variance
    const double Noise_variance = CalImageNoiseVariance(resized_image, flag_boost);
    CurNoiseInfo = Noise_variance * resize_factor;

    // 4. Compute Image Quality Value based on 1~3 values.
    const double Image_Quality_Value = alpha * Gradient_value + (1 - alpha) * Entropy_value - beta * Noise_variance;
    // double Image_Quality_Value = Noise_variance;

    return Image_Quality_Value;
}
