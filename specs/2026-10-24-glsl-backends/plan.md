# Plan

1. Wire backend selection, presets and shared shader compilation.
2. Implement headless EGL/OpenGL and Vulkan host paths with a shared shader and harness.
3. Build/run both, check backend conflict rejection and existing OpenCL regression, record limitations and timings, synchronize docs.

All three phases completed locally on 2026-10-24. Initial coverage intentionally
remains sigmoid; raw baseline evidence and the unprofiled Vulkan four-element
regression are recorded in validation.md. Constitution additions are limited to
the new backend boundaries and roadmap entry.
