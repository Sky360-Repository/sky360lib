#include <Halide.h>
#include <halide_trace_config.h>

namespace
{
    class WMVMonoThreshold : public Halide::Generator<WMVMonoThreshold>
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
            Var x, y;
            Func i, m, v, r;
            
            Expr w[] = {w0, w1, w2};
            i(x, y) = {cast<float>(input0(x, y)), cast<float>(input1(x, y)), cast<float>(input2(x, y))};
            m(x, y) = (i(x, y)[0] * w[0]) + (i(x, y)[1] * w[1]) + (i(x, y)[2] * w[2]);
            v(x, y) = {i(x, y)[0] - m(x, y), i(x, y)[1] - m(x, y), i(x, y)[2] - m(x, y)};
            r(x, y) = (v(x, y)[0] * v(x, y)[0] * w[0]) + (v(x, y)[1] * v(x, y)[1] * w[1]) + (v(x, y)[2] * v(x, y)[2] * w[2]);
            output(x, y) = cast<uint8_t>(select(r(x, y) > t2, 255, 0));

            input0.set_estimates({{0, 2880}, {0, 2880}});
            input1.set_estimates({{0, 2880}, {0, 2880}});
            input2.set_estimates({{0, 2880}, {0, 2880}});
            w0.set_estimate(0.3333f);
            w1.set_estimate(0.3333f);
            w2.set_estimate(0.3333f);
            t2.set_estimate(225.0f);
            output.set_estimates({{0, 2880}, {0, 2880}});

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

HALIDE_REGISTER_GENERATOR(WMVMonoThreshold, wmv_mono_threshold)
