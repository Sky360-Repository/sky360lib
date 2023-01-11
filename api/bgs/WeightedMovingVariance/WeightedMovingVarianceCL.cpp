#include "WeightedMovingVarianceCL.hpp"

// opencv legacy includes
// #include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

using namespace sky360lib::bgs;

WeightedMovingVarianceCL::WeightedMovingVarianceCL(const WeightedMovingVarianceParams& _params)
    : CoreBgs(1),
      m_params(_params)
{
    initOpenCL();
}

WeightedMovingVarianceCL::~WeightedMovingVarianceCL()
{
    clearCL();
}

void WeightedMovingVarianceCL::getBackgroundImage(cv::Mat &)
{
}

void WeightedMovingVarianceCL::clearCL()
{
    for (size_t i = 0; i < m_numProcessesParallel; ++i)
    {
        // if (imgInputPrev[i].pImgOutputCuda != nullptr)
        //     cudaFree(imgInputPrev[i].pImgOutputCuda);
        // if (imgInputPrev[i].pImgMem[0] != nullptr)
        //     cudaFree(imgInputPrev[i].pImgMem[0]);
        // if (imgInputPrev[i].pImgMem[1] != nullptr)
        //     cudaFree(imgInputPrev[i].pImgMem[1]);
        // if (imgInputPrev[i].pImgMem[2] != nullptr)
        //     cudaFree(imgInputPrev[i].pImgMem[2]);
    }
}

void WeightedMovingVarianceCL::initOpenCL()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cout << "No OpenCL supported." << std::endl;
        return;
    }
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cout << "OpenCL Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "OpenCL Number of devices: " << devices.size() << std::endl;
    m_device = devices[0];

    // output some device configuration
    cl_uint computeUnits;
    m_device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnits);
    std::cout << "Using device: " << m_device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "OpenCL Number of Compute Units: " << computeUnits << std::endl;

    // Create the context and command queue
    m_context = cl::Context({m_device});
    m_queue = cl::CommandQueue(m_context, m_device);

    // Create the program and kernel
    std::string kernelSource = R"(
void kernel monoThreshold(global const uchar * i1, global const uchar *i2, global const uchar *i3,
                          global uchar *o, global float* _w, float _thresholdSquared)
{
    const int gid = get_global_id(0);
    const float dI[3] = {(float)i1[gid], (float)i2[gid], (float)i3[gid]};
    const float mean = (dI[0] * _w[0]) + (dI[1] * _w[1]) + (dI[2] * _w[2]);
    const float value[3] = {dI[0] - mean, dI[1] - mean, dI[2] - mean};
    const float result = ((value[0] * value[0]) * _w[0]) + ((value[1] * value[1]) * _w[1]) + ((value[2] * value[2]) * _w[2]);
    o[gid] = result > _thresholdSquared ? 255 : 0;
}
)";
    m_program = cl::Program(m_context, kernelSource);
    if (m_program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " OpenCL Error building: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << std::endl;
        exit(1);
    }
    m_wmvKernel = cl::Kernel(m_program, "monoThreshold");
}

void WeightedMovingVarianceCL::initialize(const cv::Mat &)
{
    imgInputPrev.resize(m_numProcessesParallel);
    for (size_t i = 0; i < m_numProcessesParallel; ++i)
    {
        imgInputPrev[i].currentRollingIdx = 0;
        imgInputPrev[i].firstPhase = 0;
        imgInputPrev[i].pImgSize = m_imgSizesParallel[i].get();
        imgInputPrev[i].pImgInput = nullptr;
        imgInputPrev[i].pImgInputPrev1 = nullptr;
        imgInputPrev[i].pImgInputPrev2 = nullptr;

        imgInputPrev[i].pImgMem[0] = cl::Buffer(m_context, CL_MEM_READ_ONLY, imgInputPrev[i].pImgSize->size);
        imgInputPrev[i].pImgMem[1] = cl::Buffer(m_context, CL_MEM_READ_ONLY, imgInputPrev[i].pImgSize->size);
        imgInputPrev[i].pImgMem[2] = cl::Buffer(m_context, CL_MEM_READ_ONLY, imgInputPrev[i].pImgSize->size);
        imgInputPrev[i].bImgOutput = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, imgInputPrev[i].pImgSize->numPixels);
        imgInputPrev[i].bWeight = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), (void*)m_params.weight);
        rollImages(imgInputPrev[i]);
    }
}

void WeightedMovingVarianceCL::rollImages(RollingImages &rollingImages)
{
    const auto rollingIdx = ROLLING_BG_IDX[rollingImages.currentRollingIdx % 3];
    rollingImages.pImgInput = &rollingImages.pImgMem[rollingIdx[0]];
    rollingImages.pImgInputPrev1 = &rollingImages.pImgMem[rollingIdx[1]];
    rollingImages.pImgInputPrev2 = &rollingImages.pImgMem[rollingIdx[2]];

    ++rollingImages.currentRollingIdx;
}

void WeightedMovingVarianceCL::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput, int _numProcess)
{
    if (_imgOutput.empty())
    {
        _imgOutput.create(_imgInput.size(), CV_8UC1);
    }
    process(_imgInput, _imgOutput, imgInputPrev[_numProcess]);
    rollImages(imgInputPrev[_numProcess]);
}

void WeightedMovingVarianceCL::process(const cv::Mat &_imgInput,
                                         cv::Mat &_imgOutput,
                                         RollingImages &_imgInputPrev)
{
    const size_t numPixels = _imgInput.size().area();
    //memcpy(_imgInputPrev.pImgInput, _imgInput.data, _imgInputPrev.pImgSize->size);
    m_queue.enqueueWriteBuffer(*_imgInputPrev.pImgInput, CL_TRUE, 0, _imgInputPrev.pImgSize->size, _imgInput.data);

    if (_imgInputPrev.firstPhase < 2)
    { 
        ++_imgInputPrev.firstPhase;
        return;
    }

    // Set the kernel arguments and run the kernel
    m_wmvKernel.setArg(0, *_imgInputPrev.pImgInput);
    m_wmvKernel.setArg(1, *_imgInputPrev.pImgInputPrev1);
    m_wmvKernel.setArg(2, *_imgInputPrev.pImgInputPrev2);
    m_wmvKernel.setArg(3, _imgInputPrev.bImgOutput);
    m_wmvKernel.setArg(4, _imgInputPrev.bWeight);
    m_wmvKernel.setArg(5, m_params.thresholdSquared);

    m_queue.enqueueNDRangeKernel(m_wmvKernel, cl::NullRange, cl::NDRange(numPixels), cl::NullRange);

    // Copy the result from the device to the host
    m_queue.enqueueReadBuffer(_imgInputPrev.bImgOutput, CL_TRUE, 0, numPixels, _imgOutput.data);
}

void testOpenCL()
{
    // Get the platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cout << "No OpenCL supported." << std::endl;
        return;
    }
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    // Print the device type
    cl_device_type deviceType;
    device.getInfo(CL_DEVICE_TYPE, &deviceType);
    if (deviceType & CL_DEVICE_TYPE_GPU)
    {
        std::cout << "Running on GPU" << std::endl;
    }
    else if (deviceType & CL_DEVICE_TYPE_CPU)
    {
        std::cout << "Running on CPU" << std::endl;
    }
    else
    {
        std::cout << "Running on unknown device type" << std::endl;
    }

    // Create the context and command queue
    cl::Context context({device});
    cl::CommandQueue queue(context, device);

    // Create the program and kernel
    std::string kernelSource = R"(
    __kernel void add(__global int* a, __global int* b, __global int* c) {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
  )";
    cl::Program program(context, kernelSource);
    program.build({device});
    cl::Kernel add(program, "add");

    // Create the input and output arrays
    const int N = 1024;
    int a[N], b[N], c[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }
    cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, a);
    cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, b);
    cl::Buffer C(context, CL_MEM_WRITE_ONLY, sizeof(int) * N);

    // Set the kernel arguments and run the kernel
    add.setArg(0, A);
    add.setArg(1, B);
    add.setArg(2, C);
    queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(N), cl::NullRange);

    // Copy the result from the device to the host
    queue.enqueueReadBuffer(C, CL_TRUE, 0, sizeof(int) * N, c);

    // Check the result
    bool correct = true;
    for (int i = 0; i < N; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            correct = false;
            break;
        }
    }
    std::cout << (correct ? "Success" : "Failure") << std::endl;
}
