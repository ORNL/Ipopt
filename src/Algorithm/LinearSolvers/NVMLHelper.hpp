#pragma once

#include <iostream>
#include <nvml.h>

class NVMLHelper
{
public:
  static int getAvailableGPUMemory()
  {
    nvmlReturn_t result;
    unsigned int device_count, i;
    nvmlDevice_t device;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
      printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
      return 1;
    }

    // Get the number of GPUs
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS)
    {
      printf("Failed to get device count: %s\n", nvmlErrorString(result));
      nvmlShutdown();
      return 1;
    }

    // Loop through each GPU and get its memory information
    for (i = 0; i < device_count; i++)
    {
      result = nvmlDeviceGetHandleByIndex(i, &device);
      if (result != NVML_SUCCESS)
      {
        printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(result));
        nvmlShutdown();
        return 1;
      }

      // Get memory information
      nvmlMemory_t memory_info;
      result = nvmlDeviceGetMemoryInfo(device, &memory_info);
      if (result != NVML_SUCCESS)
      {
        printf("Failed to get memory info for device %d: %s\n", i, nvmlErrorString(result));
        nvmlShutdown();
        return 1;
      }

      // Print total, free, and used memory
      printf("GPU %d:\n", i);
      printf("  Total memory: %ld\n", memory_info.total / (1024 * 1024));
      printf("  Free memory: %ld\n", memory_info.free / (1024 * 1024));
      printf("  Used memory: %ld\n", memory_info.used / (1024 * 1024));
    }

    // Shutdown NVML
    result = nvmlShutdown();
    if (result != NVML_SUCCESS)
    {
      printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
      return 1;
    }

    return 0;
  }
};