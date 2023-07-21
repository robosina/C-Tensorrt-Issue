import os
import time

import cv2
import requests
import numpy as np
from PIL import Image
from numpy.testing import suppress_warnings
import infery
from onnxsim import simplify
import onnx
import warnings
from sklearn.preprocessing import normalize

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import logging

if __name__ == '__main__':
    # Set up logging
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG

    # Create a file handler
    log_path = "export_log.txt"  # Your log file path
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG for the file handler

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG for the console handler

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    def predict_with_infery(onnx_file_name, file_name, batch_size):
        image = cv2.imread(file_name)
        image = (image - 127.5) * 0.0078125
        image = np.expand_dims(image, 0).astype(np.float32)
        device = 'cpu'  # gpu needs pycuda(I didn't install it since we don't need it in onnx layer)
        logger.info(f'device type is:{device} and onnx model is {onnx_file_name}')
        onnx_model = infery.load(model_path=onnx_file_name, framework_type='onnx', inference_hardware=device)
        embedding = onnx_model.predict(image)
        embedding = np.array(embedding).reshape(1, -1)
        emb_normalize = normalize(embedding)
        return emb_normalize, np.concatenate([embedding, emb_normalize], 0)


    def print_model_bindings(model_path: str):
        # Load nmsed model to show input output bindings
        model = onnx.load(model_path)

        logger.info('--------------------')
        logger.info('Input bindings is:')
        # Get the name and shape of each input in the model
        for input_tensor in model.graph.input:
            logger.info(f'{input_tensor.name}, {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}')

        logger.info('--------------------')
        logger.info('Output bindings is:')
        # Get the name and shape of each output in the model
        for output_tensor in model.graph.output:
            output_tensor_log = f'{output_tensor.name}  {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}'
            logger.info(output_tensor_log)


    import numpy as np
    import onnx

    model_name = './Data/Example.onnx'
    print_model_bindings(model_name)

    # Load your ONNX model
    model = onnx.load(model_name)

    # Go through every node (operation) in the graph
    for node in model.graph.node:
        # Go through every attribute of the node
        for attr in node.attribute:
            # If the attribute is of type INTS and its field is not empty
            if attr.ints and attr.name == 'axes':
                # Convert the list of ints from int64 to int32
                attr.ints[:] = np.array(attr.ints, dtype=np.int32)

    # Save the modified ONNX model
    onnx.save(model, 'GhostFaceNet_W1_simplified_f32.onnx')
    device = 'cpu'  # gpu needs pycuda(I didn't install it since we don't need it in onnx layer)
    logger.info(f'device type is:{device} and onnx model is {model_name}')
    onnx_model = infery.load(model_path=model_name, framework_type='onnx', inference_hardware=device)
    image = np.ones((1, 112, 112, 3), np.float32) * -0.99609375
    embedding = onnx_model.predict(image)
    embedding = np.array(embedding).reshape(1, -1)
    # print first 10 elements
    for i in range(10):
        print(embedding[0][i])
