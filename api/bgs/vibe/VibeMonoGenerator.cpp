#include <Halide.h>

#include <vector>
namespace
{
    using namespace Halide;
    class VibeMono : public Halide::Generator<VibeMono>
    {
    public:
        // The input image
        Input<Buffer<uint8_t>> image{"image", 2};
        // The background images
        Input<Buffer<uint8_t, 3>> bgImg{"bgImg"};
        // The foreground mask
        Output<Buffer<uint8_t>> fgmask{"fgmask", 2};
        // The Vibe parameters
        Input<uint64_t> nRequiredBGSamples{"nRequiredBGSamples"};
        Input<uint64_t> nBGSamples{"nBGSamples"};
        Input<uint64_t> nColorDistThreshold{"nColorDistThreshold"};
        Input<uint32_t> andLearningRate{"andLearningRate"};

        void generate()
        {
            // Create variables for the x and y coordinates of the current pixel
            Var x("x"), y("y");

            // Initialize the foreground mask to zero
            fgmask(x, y) = 0;

            // Loop over all pixels in the image
            RDom r(0, image.width(), 0, image.height());
            fgmask(r.x, r.y) = select(
                // If the number of good samples is less than the required number of
                // background samples, set the foreground mask to 255 (i.e. foreground)
                nGoodSamplesCount(r.x, r.y) < nRequiredBGSamples,
                UCHAR_MAX,
                // Otherwise, update the random background image and set the foreground
                // mask to 0 (i.e. background)
                updateBgImg(r.x, r.y));
        }

    private:
        // Calculates the L1 distance between two pixels
        Expr L1dist(Expr p1, Expr p2)
        {
            return abs(p1 - p2);
        }

        // Calculates the number of good samples at the given coordinates
        Expr nGoodSamplesCount(Expr x, Expr y)
        {
            // Initialize the good sample count to zero
            Expr count = 0;
            // Loop over all samples
            RDom r(0, nBGSamples);
            // If the L1 distance between the current pixel and the sample is less
            // than the color distance threshold, increment the good sample count
            count += select(L1dist(image(x, y), bgImg(r, x, y)) < nColorDistThreshold, 1, 0);
            return count;
        }

        // Updates a random background image and returns 0
        Expr updateBgImg(Expr x, Expr y)
        {
            // Choose a random background image
            Expr bgIdx = random_uint() & andLearningRate;
            // Update the pixel value in the chosen background image
            bgImg(bgIdx, x, y) = image(x, y);

            // Choose a random neighbor position
            Expr nx, ny;
            //getNeighborPosition_3x3(x, y, image.width(), image.height(), random_uint(), &nx, &ny);
            // Choose a random background image
            Expr bgIdx2 = random_uint() & andLearningRate;
            // Update the pixel value in the chosen background image
            bgImg(bgIdx2, nx, ny) = image(x, y);

            return 0;
        }
    };
} // namespace

HALIDE_REGISTER_GENERATOR(VibeMono, vibe_mono)
