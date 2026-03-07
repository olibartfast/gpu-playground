#pragma once
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CL_CHECK(call)                                                        \
  do {                                                                        \
    cl_int cl_err__ = (call);                                                 \
    if (cl_err__ != CL_SUCCESS) {                                             \
      fprintf(stderr, "OpenCL error %s:%d: code %d\n",                       \
              __FILE__, __LINE__, cl_err__);                                  \
      std::abort();                                                            \
    }                                                                         \
  } while (0)

// Return the first available GPU device and populate ctx and queue.
// Aborts if no GPU platform/device is found.
inline cl_device_id clSetupGPU(cl_context& ctx, cl_command_queue& queue) {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        std::abort();
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    cl_device_id device = nullptr;
    for (auto& plat : platforms) {
        cl_uint num_dev = 0;
        cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_dev);
        if (err != CL_SUCCESS || num_dev == 0) continue;
        std::vector<cl_device_id> devs(num_dev);
        CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_dev, devs.data(), nullptr));
        device = devs[0];
        break;
    }
    if (!device) {
        fprintf(stderr, "No OpenCL GPU device found\n");
        std::abort();
    }

    cl_int err;
    ctx   = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err);
    return device;
}

// Compile an OpenCL program from source.  Prints build log and aborts on error.
inline cl_program clBuildFromSource(cl_context ctx, cl_device_id dev, const char* src) {
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        fprintf(stderr, "OpenCL build error:\n%s\n", log.c_str());
        std::abort();
    }
    return prog;
}

// Release all OpenCL objects created by clSetupGPU / clBuildFromSource.
inline void clTeardown(cl_context ctx, cl_command_queue queue,
                       cl_program prog, cl_kernel kernel) {
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}
