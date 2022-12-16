#include <Halide.h>
#include <halide_trace_config.h>

namespace
{
    class WMVColorThresholdGenerator : public Halide::Generator<WMVColorThresholdGenerator>
    {
    public:
        Input<Buffer<uint8_t, 3>> input0{"input1"};
        Input<Buffer<uint8_t, 3>> input1{"input2"};
        Input<Buffer<uint8_t, 3>> input2{"input3"};
        Input<float> w0{"w0"};
        Input<float> w1{"w1"};
        Input<float> w2{"w2"};
        Input<float> t2{"t2"};
        Output<Buffer<uint8_t, 2>> output{"output"};

        void generate()
        {
            Var x, y;

            Func iR, iG, iB;
            Func mR, mG, mB;
            Func vR, vG, vB;
            Func r, g, b;
            Func result;
            Expr w[] = {w0, w1, w2};
            iR(x, y) = {cast<float>(input0(x, y, 0)), cast<float>(input1(x, y, 0)), cast<float>(input2(x, y, 0))};
            iG(x, y) = {cast<float>(input0(x, y, 1)), cast<float>(input1(x, y, 1)), cast<float>(input2(x, y, 1))};
            iB(x, y) = {cast<float>(input0(x, y, 2)), cast<float>(input1(x, y, 2)), cast<float>(input2(x, y, 2))};
            mR(x, y) = (iR(x, y)[0] * w[0]) + (iR(x, y)[1] * w[1]) + (iR(x, y)[2] * w[2]);
            mG(x, y) = (iG(x, y)[0] * w[0]) + (iG(x, y)[1] * w[1]) + (iG(x, y)[2] * w[2]);
            mB(x, y) = (iB(x, y)[0] * w[0]) + (iB(x, y)[1] * w[1]) + (iB(x, y)[2] * w[2]);
            vR(x, y) = {iR(x, y)[0] - mR(x, y), iR(x, y)[1] - mR(x, y), iR(x, y)[2] - mR(x, y)};
            vG(x, y) = {iG(x, y)[0] - mG(x, y), iG(x, y)[1] - mG(x, y), iG(x, y)[2] - mG(x, y)};
            vB(x, y) = {iB(x, y)[0] - mB(x, y), iB(x, y)[1] - mB(x, y), iB(x, y)[2] - mB(x, y)};
            r(x, y) = (vR(x, y)[0] * vR(x, y)[0] * w[0]) + (vR(x, y)[1] * vR(x, y)[1] * w[1]) + (vR(x, y)[2] * vR(x, y)[2] * w[2]);
            g(x, y) = (vG(x, y)[0] * vG(x, y)[0] * w[0]) + (vG(x, y)[1] * vG(x, y)[1] * w[1]) + (vG(x, y)[2] * vG(x, y)[2] * w[2]);
            b(x, y) = (vB(x, y)[0] * vB(x, y)[0] * w[0]) + (vB(x, y)[1] * vB(x, y)[1] * w[1]) + (vB(x, y)[2] * vB(x, y)[2] * w[2]);
            result(x, y) = r(x, y) * 0.299f + g(x, y) * 0.587f + b(x, y) * 0.114f;
            output(x, y) = cast<uint8_t>(select(result(x, y) > t2, 255, 0));

            input0.set_estimates({{0, 2880}, {0, 2880}, {0, 2}});
            input1.set_estimates({{0, 2880}, {0, 2880}, {0, 2}});
            input2.set_estimates({{0, 2880}, {0, 2880}, {0, 2}});
            w0.set_estimate(0.3333f);
            w1.set_estimate(0.3333f);
            w2.set_estimate(0.3333f);
            t2.set_estimate(225.0f);
            output.set_estimates({{0, 2880}, {0, 2880}});

            input0.dim(0).set_stride(3);
            input0.dim(2).set_stride(1);
            input1.dim(0).set_stride(3);
            input1.dim(2).set_stride(1);
            input2.dim(0).set_stride(3);
            input2.dim(2).set_stride(1);

            if (!get_auto_schedule())
            {
                if (get_target().has_gpu_feature()) 
                {
                    Var xi("xi"), yi("yi");
                    output.compute_root()
                        .gpu_tile(x, y, xi, yi, 32, 8);
                }
                else
                {
                    output.compute_root()
                        .parallel(y)
                        .vectorize(x, 8);
                }
            }
        }
    };

} // namespace

HALIDE_REGISTER_GENERATOR(WMVColorThresholdGenerator, wmv_color_threshold)
