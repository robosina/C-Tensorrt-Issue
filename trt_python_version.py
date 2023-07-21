import os.path

# add a path to search list
import sys
sys.path.append('C:/SMS_prerequisite/TensorRT-8.6.0.12/lib')
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import cv2

# main guard
if __name__ == '__main__':
    engine_path = "Data/Example.engine"
    # Load your ONNX model.
    onnx_model = onnx.load("Data/Example.onnx")

    # Create a TensorRT logger and builder.
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    # Enable fp16 precision if the platform supports it.
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create a TensorRT network object with EXPLICIT_BATCH flag.
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    # Parse the ONNX model to get the network.
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_model.SerializeToString()):
        print('Failed to parse the ONNX model. Check the error log.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    if not os.path.exists(engine_path):
        # Build the engine.
        engine = builder.build_engine(network, config)

        # Save the engine to a file.
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
    else:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

    # data feed, I've supposed that all of the image pixels after preprocessing is -0.99609375
    image = np.ones((1, 112, 112, 3), np.float32) * -0.99609375

    # Run the inference.
    with engine.create_execution_context() as context:
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Allocate device memory for inputs and outputs.
        d_inputs = []
        d_outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers.
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                d_inputs.append((host_mem, device_mem))
            else:
                d_outputs.append((host_mem, device_mem))

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_inputs[0][1], image, stream)

        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(d_outputs[0][0], d_outputs[0][1], stream)

        # Synchronize the stream.
        stream.synchronize()

        # Our prediction will be the output from the inference.
        prediction = d_outputs[0][0]
        first_batch = d_outputs[0][0]  # we only have one batch

        # print first 10 results
        for i in range(10):
            print(f'Python {i}th result is:{first_batch[i]}')