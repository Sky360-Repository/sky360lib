#include <Halide.h>
#include <halide_trace_config.h>

namespace
{
    class WMVHalide : public Halide::Generator<WMVHalide>
    {
    public:
        Input<Buffer<uint8_t, 2>> input0{"input1"};
        Input<Buffer<uint8_t, 2>> input1{"input2"};
        Input<Buffer<uint8_t, 2>> input2{"input3"};
        Input<float> w0{"w0"};
        Input<float> w1{"w1"};
        Input<float> w2{"w2"};
        Input<float> t2{"t2"};
        Output<Buffer<uint8_t, 2>> output{"output"};

        void generate()
        {
            Var x("x"), y("y");
            // const float dI[]{(float)*i1, (float)*i2, (float)*i3};
            // const float mean{(dI[0] * _params.weight[0]) + (dI[1] * _params.weight[1]) + (dI[2] * _params.weight[2])};
            // const float value[]{dI[0] - mean, dI[1] - mean, dI[2] - mean};
            // const float result{((value[0] * value[0]) * _params.weight[0]) 
            //                     + ((value[1] * value[1]) * _params.weight[1]) 
            //                     + ((value[2] * value[2]) * _params.weight[2])};
            // *o = result > _params.thresholdSquared ? UCHAR_MAX : ZERO_UC;
            //Func mean("mean");
            Expr i0 = cast<float>(input0(x, y));
            Expr i1 = cast<float>(input1(x, y));
            Expr i2 = cast<float>(input2(x, y));
            Expr mean = (i0 * w0) + (i1 * w1) + (i2 * w2);
            Expr v0 = i0 - mean;
            Expr v1 = i1 - mean;
            Expr v2 = i2 - mean;
            Expr result = (v0 * v0 * w0) + (v1 * v1 * w1) + (v2 * v2 * w2);

            input0.set_estimates({{0, 2880}, {0, 2880}});
            input1.set_estimates({{0, 2880}, {0, 2880}});
            input2.set_estimates({{0, 2880}, {0, 2880}});
            w0.set_estimate(0.3333f);
            w1.set_estimate(0.3333f);
            w2.set_estimate(0.3333f);
            t2.set_estimate(32000.0f);

            if (!get_auto_schedule())
            {
                if (get_target().has_gpu_feature()) 
                {
                    output(x, y) = cast<uint8_t>(select(result > t2, 255, 0));
                    output.set_estimates({{0, 2880}, {0, 2880}});
                    Var xi("xi"), yi("yi");
                    output.compute_root()
                        .gpu_tile(x, y, xi, yi, 32, 8);
                }
                else
                {
                    output(x, y) = cast<uint8_t>(select(result > t2, 0, 255));
                    output.set_estimates({{0, 2880}, {0, 2880}});
                    output.compute_root()
                        .parallel(y)
                        .vectorize(x, 8);
                }
            }
            else
            {
                output(x, y) = cast<uint8_t>(select(result > t2, 128, 128));
                output.set_estimates({{0, 2880}, {0, 2880}});
            }
        }
    };

} // namespace

HALIDE_REGISTER_GENERATOR(WMVHalide, wmv_halide)
