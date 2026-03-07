#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Return the first available GPU device.
// Throws std::runtime_error if no GPU platform/device is found.
inline cl::Device clppGetGPUDevice() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        try {
            p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        } catch (const cl::Error&) { continue; }
        if (!devices.empty()) return devices[0];
    }
    throw std::runtime_error("No OpenCL GPU device found");
}

// Compile an OpenCL program from source.
// Prints build log and calls abort() on compile error.
// Pass compiler options such as "-D TILE_SIZE=16" via options.
inline cl::Program clppBuildProgram(const cl::Context& ctx,
                                     const cl::Device& dev,
                                     const std::string& src,
                                     const char* options = nullptr) {
    cl::Program prog(ctx, src);
    try {
        prog.build({dev}, options);
    } catch (const cl::Error&) {
        std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        fprintf(stderr, "OpenCL build error:\n%s\n", log.c_str());
        std::abort();
    }
    return prog;
}

// Return the preferred work-group size aligned to the kernel's preferred multiple,
// clamped to the kernel's maximum work-group size and the 'target' hint (default 256).
inline size_t clppPreferredLocalSize(const cl::Kernel& kernel,
                                      const cl::Device& dev,
                                      size_t target = 256) {
    size_t preferred_mult =
        kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);
    size_t max_wgs =
        kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev);
    if (preferred_mult == 0) preferred_mult = 64;
    size_t local = (target / preferred_mult) * preferred_mult;
    if (local == 0) local = preferred_mult;
    if (max_wgs > 0 && local > max_wgs) local = max_wgs;
    return local;
}
