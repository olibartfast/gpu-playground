#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CL_CHECK(err) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d\n", (int)(err), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

// Find first GPU device, create context and command queue.
inline cl_device_id clSetupGPU(cl_context& ctx, cl_command_queue& queue) {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    cl_device_id dev = nullptr;
    for (auto& p : platforms) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS) continue;
        if (num_devices == 0) continue;
        std::vector<cl_device_id> devices(num_devices);
        CL_CHECK(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr));
        dev = devices[0];
        break;
    }
    if (!dev) { fprintf(stderr, "No OpenCL GPU device found\n"); std::abort(); }

    cl_int err;
    ctx   = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CL_CHECK(err);
    queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err); CL_CHECK(err);
    return dev;
}

// Build program from source string; prints build log and aborts on compile error.
inline cl_program clBuildFromSource(cl_context ctx, cl_device_id dev,
                                    const char* src, const char* options = nullptr) {
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err); CL_CHECK(err);
    if (clBuildProgram(prog, 1, &dev, options, nullptr, nullptr) != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        fprintf(stderr, "OpenCL build error:\n%s\n", log.c_str());
        std::abort();
    }
    return prog;
}

// Release all OpenCL resources.
inline void clTeardown(cl_context ctx, cl_command_queue queue,
                       cl_program prog, cl_kernel kernel) {
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}

// Return the preferred work-group size aligned to the kernel's preferred multiple,
// clamped to the kernel's maximum work-group size and the 'target' hint (default 256).
inline size_t clPreferredLocalSize(cl_kernel kernel, cl_device_id dev, size_t target = 256) {
    size_t preferred_mult = 0, max_wgs = 0;
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(size_t), &preferred_mult, nullptr);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &max_wgs, nullptr);
    if (preferred_mult == 0) preferred_mult = 64;
    size_t local = (target / preferred_mult) * preferred_mult;
    if (local == 0) local = preferred_mult;
    if (max_wgs > 0 && local > max_wgs) local = max_wgs;
    return local;
}
