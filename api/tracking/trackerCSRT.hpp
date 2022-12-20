#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace sky360lib::tracking
{
    using namespace cv;
    /** @brief the CSRT tracker

    The implementation is based on @cite Lukezic_IJCV2018 Discriminative Correlation Filter with Channel and Spatial Reliability
    */
    class TrackerCSRT : public Tracker
    {
    protected:
        TrackerCSRT(); // use ::create()
    public:
        virtual ~TrackerCSRT() override;

        struct Params
        {
            Params();

            bool use_hog;
            bool use_color_names;
            bool use_gray;
            bool use_rgb;
            bool use_channel_weights;
            bool use_segmentation;

            std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
            float kaiser_alpha;
            float cheb_attenuation;

            float template_size;
            float gsl_sigma;
            float hog_orientations;
            float hog_clip;
            float padding;
            float filter_lr;
            float weights_lr;
            int num_hog_channels_used;
            int admm_iterations;
            int histogram_bins;
            float histogram_lr;
            int background_ratio;
            int number_of_scales;
            float scale_sigma_factor;
            float scale_model_max_area;
            float scale_lr;
            float scale_step;

            float psr_threshold; //!< we lost the target, if the psr is lower than this.
        };

        static Ptr<TrackerCSRT> create(const TrackerCSRT::Params &parameters = TrackerCSRT::Params());

        virtual void setInitialMask(InputArray mask) = 0;
    };
}