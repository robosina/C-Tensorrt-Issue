#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/device_vector.h>
#include <thrust/fill.h>

//cuda headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace nvinfer1;

// CUDA Kernel to fill an array with a specific value
__global__ void fillKernel(float* devPtr, const float val, const int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        devPtr[idx] = val;
}

// Function to launch the kernel
void fillWith(float* devPtr, float val, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    fillKernel<<<numBlocks, blockSize>>>(devPtr, val, N);
    cudaDeviceSynchronize();
}

// Simple Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // suppress info-level messages
        std::cout << msg << std::endl;
    }
} gLogger;

int main() {
    std::string engine_file = "Example.engine";

    // Create a TensorRT runtime
    IRuntime *runtime = createInferRuntime(gLogger);

    // Read the engine file
    std::ifstream engineStream(engine_file, std::ios::binary);
    std::string engineString((std::istreambuf_iterator<char>(engineStream)), std::istreambuf_iterator<char>());
    engineStream.close();

    // Deserialize the engine
    ICudaEngine *engine = runtime->deserializeCudaEngine(engineString.data(), engineString.size(), nullptr);

    // Create an execution context
    IExecutionContext *context = engine->createExecutionContext();

    // Allocate memory on the GPU for the input and output data
    float *input_data;
    float *output_data;
    int input_size = 1 * 112 * 112 * 3;
    int output_size = 512;
    cudaMalloc((void **) &input_data, input_size * sizeof(float));
    cudaMalloc((void **) &output_data, output_size * sizeof(float));

    // Set the input data
    float input_value = -0.49609375;
//    cudaMemset(input_data, input_value, input_size * sizeof(float));

    thrust::device_ptr<float> th_input_data(input_data);
    thrust::fill(th_input_data, th_input_data + input_size, input_value);

    // Set up the execution bindings
    void *bindings[2] = {input_data, output_data};

    // Run inference
    context->executeV2(bindings);

    // Copy the output data back to the host
    float *host_output = new float[output_size];
    cudaMemcpy(host_output, output_data, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output data
    for (int i = 0; i < 10; ++i) {
        std::cout << host_output[i] << " \n";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(input_data);
    cudaFree(output_data);
    delete[] host_output;
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}