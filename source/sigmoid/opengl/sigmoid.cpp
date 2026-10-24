#define GL_GLEXT_PROTOTYPES
#include "../glsl/sigmoid.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
void eglCheck(bool ok, const char *operation) {
  if (!ok)
    throw std::runtime_error(std::string(operation) + " EGL error " +
                             std::to_string(eglGetError()));
}
void glCheck() {
  auto error = glGetError();
  if (error != GL_NO_ERROR)
    throw std::runtime_error("OpenGL error " + std::to_string(error));
}
struct Context {
  EGLDisplay display = EGL_NO_DISPLAY;
  EGLContext context = EGL_NO_CONTEXT;
  GLuint programs[2] = {};
  Context() {
    try {
      // Mesa's surfaceless platform needs no display server or window.
      auto platform = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
          eglGetProcAddress("eglGetPlatformDisplayEXT"));
      if (platform)
        display = platform(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY,
                           nullptr);
      EGLint major, minor;
      if (display == EGL_NO_DISPLAY ||
          !eglInitialize(display, &major, &minor)) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglCheck(eglInitialize(display, &major, &minor), "eglInitialize");
      }
      eglCheck(eglBindAPI(EGL_OPENGL_API), "eglBindAPI");
      EGLint attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE,
                        EGL_PBUFFER_BIT, EGL_NONE};
      EGLConfig config;
      EGLint count = 0;
      eglCheck(eglChooseConfig(display, attrs, &config, 1, &count) && count > 0,
               "eglChooseConfig");
      EGLint ctxAttrs[] = {EGL_CONTEXT_MAJOR_VERSION,
                           4,
                           EGL_CONTEXT_MINOR_VERSION,
                           3,
                           EGL_CONTEXT_OPENGL_PROFILE_MASK,
                           EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                           EGL_NONE};
      context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttrs);
      eglCheck(context != EGL_NO_CONTEXT,
               "eglCreateContext (OpenGL 4.3 required)");
      eglCheck(eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context),
               "eglMakeCurrent (surfaceless required)");
      std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << " | "
                << glGetString(GL_VERSION) << '\n';
      std::ifstream file(GLSL_SOURCE);
      if (!file)
        throw std::runtime_error("Cannot open GLSL source: " GLSL_SOURCE);
      std::string source((std::istreambuf_iterator<char>(file)), {});
      for (int variant = 0; variant < 2; ++variant) {
        std::string text = source;
        text.insert(text.find('\n') + 1,
                    variant ? "#define WIDTH 4\n" : "#define WIDTH 1\n");
        const char *ptr = text.c_str();
        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &ptr, nullptr);
        glCompileShader(shader);
        GLint ok;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
          char log[8192] = {};
          glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
          glDeleteShader(shader);
          throw std::runtime_error(std::string("GLSL compilation: ") + log);
        }
        programs[variant] = glCreateProgram();
        glAttachShader(programs[variant], shader);
        glLinkProgram(programs[variant]);
        glDeleteShader(shader);
        glGetProgramiv(programs[variant], GL_LINK_STATUS, &ok);
        if (!ok) {
          char log[8192] = {};
          glGetProgramInfoLog(programs[variant], sizeof(log), nullptr, log);
          throw std::runtime_error(std::string("GLSL link: ") + log);
        }
      }
      glCheck();
    } catch (...) {
      release();
      throw;
    }
  }
  void release() {
    if (context != EGL_NO_CONTEXT) {
      for (GLuint program : programs)
        if (program)
          glDeleteProgram(program);
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroyContext(display, context);
    }
    if (display != EGL_NO_DISPLAY)
      eglTerminate(display);
  }
  ~Context() { release(); }
};
struct Resources {
  GLuint buffers[2] = {}, queries[2] = {};
  Resources() {
    glGenBuffers(2, buffers);
    glGenQueries(2, queries);
  }
  ~Resources() {
    glDeleteQueries(2, queries);
    glDeleteBuffers(2, buffers);
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
  Resources resources;
  GLint64 maxBytes = 0;
  glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBytes);
  GLint maxGroups;
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxGroups);
  unsigned width = variant ? 4 : 1;
  unsigned groups = (unsigned(n) + 128 * width - 1) / (128 * width);
  if (size_t(n) * sizeof(float) > size_t(maxBytes) ||
      groups > unsigned(maxGroups))
    throw std::runtime_error("Sigmoid exceeds OpenGL device limits");
  for (unsigned i = 0; i < 2; ++i) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resources.buffers[i]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size_t(n) * sizeof(float),
                 i ? nullptr : input, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, resources.buffers[i]);
  }
  glUseProgram(ctx.programs[variant]);
  glUniform1ui(glGetUniformLocation(ctx.programs[variant], "n"), unsigned(n));
  glQueryCounter(resources.queries[0], GL_TIMESTAMP);
  glDispatchCompute(groups, 1, 1);
  glQueryCounter(resources.queries[1], GL_TIMESTAMP);
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, resources.buffers[1]);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size_t(n) * sizeof(float),
                     output);
  GLuint64 start, stop;
  glGetQueryObjectui64v(resources.queries[0], GL_QUERY_RESULT, &start);
  glGetQueryObjectui64v(resources.queries[1], GL_QUERY_RESULT, &stop);
  glCheck();
  if (ms)
    *ms = float(double(stop - start) / 1e6);
}
} // namespace
void sigmoid_gpu(const float *x, float *y, int n, float *ms) {
  run(x, y, n, ms, 0);
}
void sigmoid2_gpu(const float *x, float *y, int n, float *ms) {
  run(x, y, n, ms, 1);
}
