# Compile normal project.
cd build
cmake ..
make

# Compile openacc
```
nvc++ -acc -Minfo=accel openacc.cpp -o edge_detector \
  -I/usr/include/opencv4 \
  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
```
