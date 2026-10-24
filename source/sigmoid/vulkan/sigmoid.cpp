#include "../glsl/sigmoid.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace {
struct Context {
  vk::UniqueInstance instance;
  vk::PhysicalDevice physical;
  vk::PhysicalDeviceProperties properties;
  uint32_t family = 0, timestampBits = 0;
  vk::UniqueDevice device;
  vk::Queue queue;
  vk::UniqueDescriptorSetLayout descriptorLayout;
  vk::UniquePipelineLayout pipelineLayout;
  vk::UniquePipeline pipelines[2];
  Context() {
    vk::ApplicationInfo app("gpu_playground", 1, nullptr, 0,
                            VK_API_VERSION_1_1);
    instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &app));
    // Prefer a GPU over a software ICD; log the choice for comparisons.
    int best = -1;
    for (auto candidate : instance->enumeratePhysicalDevices()) {
      auto props = candidate.getProperties();
      auto families = candidate.getQueueFamilyProperties();
      for (uint32_t i = 0; i < families.size(); ++i) {
        if (!(families[i].queueFlags & vk::QueueFlagBits::eCompute) ||
            !families[i].timestampValidBits)
          continue;
        if (!props.limits.timestampComputeAndGraphics ||
            props.limits.maxComputeWorkGroupInvocations < 128 ||
            props.limits.maxComputeWorkGroupSize[0] < 128)
          continue;
        int score = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? 3
                    : props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu
                        ? 2
                        : 1;
        if (score > best) {
          best = score;
          physical = candidate;
          properties = props;
          family = i;
          timestampBits = families[i].timestampValidBits;
        }
      }
    }
    if (best < 0)
      throw std::runtime_error(
          "No Vulkan compute device with timestamps and 128-thread workgroups");
    std::cout << "Vulkan device: " << properties.deviceName.data()
              << " | driverVersion=" << properties.driverVersion
              << " | API=" << VK_VERSION_MAJOR(properties.apiVersion) << '.'
              << VK_VERSION_MINOR(properties.apiVersion) << '\n';
    float priority = 1;
    vk::DeviceQueueCreateInfo queueInfo({}, family, 1, &priority);
    device = physical.createDeviceUnique(vk::DeviceCreateInfo({}, queueInfo));
    queue = device->getQueue(family, 0);
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 2; ++i)
      bindings.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1,
                            vk::ShaderStageFlagBits::eCompute);
    descriptorLayout = device->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings));
    vk::PushConstantRange range(vk::ShaderStageFlagBits::eCompute, 0,
                                sizeof(uint32_t));
    auto layout = *descriptorLayout;
    pipelineLayout = device->createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo({}, layout, range));
    const char *paths[] = {GLSL_SPIRV_1, GLSL_SPIRV_4};
    for (int i = 0; i < 2; ++i) {
      std::ifstream file(paths[i], std::ios::binary | std::ios::ate);
      if (!file)
        throw std::runtime_error(std::string("Cannot open SPIR-V: ") +
                                 paths[i]);
      auto bytes = file.tellg();
      if (bytes <= 0 || size_t(bytes) % 4)
        throw std::runtime_error("Invalid SPIR-V size");
      std::vector<uint32_t> words(size_t(bytes) / 4);
      file.seekg(0);
      if (!file.read(reinterpret_cast<char *>(words.data()), bytes))
        throw std::runtime_error("Cannot read SPIR-V");
      auto shader = device->createShaderModuleUnique(
          vk::ShaderModuleCreateInfo({}, size_t(bytes), words.data()));
      vk::PipelineShaderStageCreateInfo stage(
          {}, vk::ShaderStageFlagBits::eCompute, *shader, "main");
      pipelines[i] =
          device
              ->createComputePipelineUnique(
                  {}, vk::ComputePipelineCreateInfo({}, stage, *pipelineLayout))
              .value;
    }
  }
};
struct Buffer {
  // Reverse destruction order: buffer before backing allocation.
  vk::UniqueDeviceMemory memory;
  vk::UniqueBuffer buffer;
  bool coherent = false;
  Buffer(Context &ctx, vk::DeviceSize bytes) {
    buffer = ctx.device->createBufferUnique(vk::BufferCreateInfo(
        {}, bytes, vk::BufferUsageFlagBits::eStorageBuffer));
    auto requirements = ctx.device->getBufferMemoryRequirements(*buffer);
    auto types = ctx.physical.getMemoryProperties();
    uint32_t index = types.memoryTypeCount;
    for (uint32_t i = 0; i < types.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (types.memoryTypes[i].propertyFlags &
           vk::MemoryPropertyFlagBits::eHostVisible)) {
        index = i;
        if (types.memoryTypes[i].propertyFlags &
            vk::MemoryPropertyFlagBits::eHostCoherent)
          break;
      }
    }
    if (index == types.memoryTypeCount)
      throw std::runtime_error("No host-visible Vulkan storage buffer memory");
    coherent = bool(types.memoryTypes[index].propertyFlags &
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    memory = ctx.device->allocateMemoryUnique(
        vk::MemoryAllocateInfo(requirements.size, index));
    ctx.device->bindBufferMemory(*buffer, *memory, 0);
  }
};
void run(const float *input, float *output, int n, float *ms, int variant) {
  if (n < 0)
    throw std::invalid_argument("Negative sigmoid size");
  if (ms)
    *ms = 0;
  if (!n)
    return;
  static Context ctx;
  auto &device = *ctx.device;
  vk::DeviceSize bytes = size_t(n) * sizeof(float);
  uint32_t width = variant ? 4 : 1;
  uint32_t groups = (uint32_t(n) + 128 * width - 1) / (128 * width);
  if (bytes > ctx.properties.limits.maxStorageBufferRange ||
      groups > ctx.properties.limits.maxComputeWorkGroupCount[0])
    throw std::runtime_error("Sigmoid exceeds Vulkan device limits");
  Buffer in(ctx, bytes), out(ctx, bytes);
  void *mapped = device.mapMemory(*in.memory, 0, VK_WHOLE_SIZE);
  std::memcpy(mapped, input, size_t(bytes));
  if (!in.coherent)
    device.flushMappedMemoryRanges(
        vk::MappedMemoryRange(*in.memory, 0, VK_WHOLE_SIZE));
  device.unmapMemory(*in.memory);
  vk::DescriptorPoolSize poolSize(vk::DescriptorType::eStorageBuffer, 2);
  auto descriptors = device.createDescriptorPoolUnique(
      vk::DescriptorPoolCreateInfo({}, 1, poolSize));
  auto layout = *ctx.descriptorLayout;
  auto set = device.allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo(*descriptors, layout))[0];
  vk::DescriptorBufferInfo infos[] = {{*in.buffer, 0, bytes},
                                      {*out.buffer, 0, bytes}};
  for (uint32_t i = 0; i < 2; ++i)
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(set, i, 0, 1, vk::DescriptorType::eStorageBuffer,
                               nullptr, &infos[i]),
        {});
  auto pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo({}, ctx.family));
  auto commands =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
          *pool, vk::CommandBufferLevel::ePrimary, 1));
  auto command = *commands[0];
  auto queries = device.createQueryPoolUnique(
      vk::QueryPoolCreateInfo({}, vk::QueryType::eTimestamp, 2));
  command.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  command.resetQueryPool(*queries, 0, 2);
  command.bindPipeline(vk::PipelineBindPoint::eCompute,
                       *ctx.pipelines[variant]);
  command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             *ctx.pipelineLayout, 0, set, {});
  uint32_t count = uint32_t(n);
  command.pushConstants(*ctx.pipelineLayout, vk::ShaderStageFlagBits::eCompute,
                        0, sizeof(count), &count);
  command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queries, 0);
  command.dispatch(groups, 1, 1);
  command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queries, 1);
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                            vk::AccessFlagBits::eHostRead);
  command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eHost, {}, barrier, {},
                          {});
  command.end();
  auto fence = device.createFenceUnique({});
  ctx.queue.submit(vk::SubmitInfo({}, {}, command), *fence);
  auto waited =
      device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
  if (waited != vk::Result::eSuccess)
    throw std::runtime_error("Vulkan fence wait failed");
  mapped = device.mapMemory(*out.memory, 0, VK_WHOLE_SIZE);
  if (!out.coherent)
    device.invalidateMappedMemoryRanges(
        vk::MappedMemoryRange(*out.memory, 0, VK_WHOLE_SIZE));
  std::memcpy(output, mapped, size_t(bytes));
  device.unmapMemory(*out.memory);
  uint64_t times[2] = {};
  auto result = device.getQueryPoolResults(
      *queries, 0, 2, sizeof(times), times, sizeof(uint64_t),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
  if (result != vk::Result::eSuccess)
    throw std::runtime_error("Vulkan timestamp query failed");
  uint64_t mask = ctx.timestampBits == 64
                      ? ~uint64_t(0)
                      : (uint64_t(1) << ctx.timestampBits) - 1;
  if (ms)
    *ms = float(double((times[1] - times[0]) & mask) *
                ctx.properties.limits.timestampPeriod / 1e6);
}
} // namespace
void sigmoid_gpu(const float *x, float *y, int n, float *ms) {
  run(x, y, n, ms, 0);
}
void sigmoid2_gpu(const float *x, float *y, int n, float *ms) {
  run(x, y, n, ms, 1);
}
