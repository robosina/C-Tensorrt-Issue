# Usage

Please Begin by using the provided Python script (`trt_python_version.py`)
to generate the TensorRT engine file. This is necessary because
our current C++ application does not possess the capability to
create this engine file independently. After that the engine has been created
you can run the cmake and it will copy the engine file beside of the exe file to use it.

The codes have been written in a simplified manner to facilitate comprehension and reproduction
by anyone. After the engine file has been created with Python,
we will then utilize this file in our C++ application.

# Python Tensorrt Version(trt_python_version.py)
```
0.23063406348228455
-0.3731584846973419
0.006619574502110481
0.10835819691419601
-0.16631896793842316
1.2145062685012817
-0.32606250047683716
0.8365474939346313
-0.8601087927818298
1.045083999633789
```

# onnx result(10 floats from 512, check_with_infery_package.py)
```
0.22878073
-0.3585716
0.0056644576
0.10903041
-0.16341211
1.2052734
-0.33571026
0.8408292
-0.85148907
1.0456058
```

# C++ Tensorrt(run cmake, cmake only needs to know the path of TensorRT, it is defined in root cmake)
```
-0.0516071
-1.07846
0.317808
0.475234
0.659989
1.37345
-0.197221
0.0824542
-0.258701
0.278051
```
