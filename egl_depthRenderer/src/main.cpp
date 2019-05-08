/*

Depth rendering codes for ScanNet 3d model with estimated poses.

Created on: May. 8, 2019
    Author: Junho Jeon, was in POSTECH, now with NaverLabs Corp.

You may need opengl extensions for compile it.
sudo apt-get install libglfw3-dev libglfw3 libglew1.5 libglew1.5-dev

*/

#include "mLibCore.h"
#include "mLibLodePNG.h"
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <common/shader.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <EGL/egl.h>
using namespace ml;
using namespace std;
using namespace cv;

static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
};    

static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
      EGL_WIDTH, pbufferWidth,
      EGL_HEIGHT, pbufferHeight,
      EGL_NONE,
};

int RenderDepthFromMesh(MeshDataf &mesh, std::string poses_path, std::string out_path, int interval)
{
  GLFWwindow* window;

  if (!glfwInit())
  {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    return -1;
  }
  int windowWidth = 640;
  int windowHeight = 480;

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Make invisible window since we render to FBO directly

  // Open a window and create its OpenGL context
  window = glfwCreateWindow(windowWidth, windowHeight, "Depth Renderer", NULL, NULL);
  if (window == NULL){
    fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  glewExperimental = true; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glViewport(0, 0, windowWidth, windowHeight);

  // Dark blue background
  glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDisable(GL_MULTISAMPLE);

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);

  GLuint depthProgramID = LoadShaders("shaders/DepthRTT.vertexshader",
                  "shaders/DepthRTT.fragmentshader", "shaders/BerycentricGeometryShader.geometryshader");
  
  // Get a handle for our "MVP" uniform
  GLuint depthMatrixID = glGetUniformLocation(depthProgramID, "depthMVP");

  // Load it into a VBO
  GLuint vertexbuffer, indexbuffer;
  glGenBuffers(1, &vertexbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
  glBufferData(GL_ARRAY_BUFFER, mesh.m_Vertices.size()*sizeof(mesh.m_Vertices[0]), &mesh.m_Vertices[0], GL_STATIC_DRAW);

  std::vector<unsigned int> indexvec;
  for (int i = 0; i < mesh.m_FaceIndicesVertices.size(); i++)
    for (int j = 0; j < 3; j++)
      indexvec.push_back(mesh.m_FaceIndicesVertices[i][j]);
  glGenBuffers(1, &indexbuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexvec.size()*sizeof(unsigned int), &indexvec[0], GL_STATIC_DRAW);

  // zNear: 0.01, zFar: 30.0 (in meters)
  float zNear = 0.01;
  float zFar = 30.0;
  ml::Matrix4x4<float> camera, proj;
  {
    float tmp[] = {
      -1.81066, 0.00000, 0.00000, 0.00000,
      0.00000, -2.41421, 0.00000, 0.00000,
      0.00000, 0.00000, -(zFar + zNear) / (zFar - zNear), -2 * zFar*zNear / (zFar-zNear),
      0.00000, 0.00000, -1.00000, 0.00000
    };
    proj = ml::Matrix4x4<float>(tmp);
  }

  GLuint FramebufferName = 0;
  glGenFramebuffers(1, &FramebufferName);
  glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

  GLuint depthTexture;
  glGenTextures(1, &depthTexture);
  glBindTexture(GL_TEXTURE_2D, depthTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);
  // No color output in the bound framebuffer, only depth.
  glDrawBuffer(GL_NONE);

  // Always check that our framebuffer is ok
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    printf("something wrong in FramebufferStatus check\n");
    return false;
  }

  // Use our shader
  glUseProgram(depthProgramID);

  int frame = 0;
  do{
    // Clear the screen
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glViewport(0, 0, windowWidth, windowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
      std::stringstream tmpFrame;
      tmpFrame << std::setw(4) << std::setfill('0') << frame;
    //   FILE *in = fopen((std::string("") + target_scene + std::string("\\pose\\frame-00") + tmpFrame.str() + ".pose.txt").c_str(), "r");
      FILE *in = fopen((poses_path + std::string("/frame-00") + tmpFrame.str() + ".pose.txt").c_str(), "r");
      if (in == NULL)
        break;
      float tmp[4][4];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          fscanf(in, "%f", &tmp[i][j]);
      fclose(in);
      if (tmp[0][0] != -1.0) {
        camera = ml::Matrix4x4<float>((float*)&tmp[0][0]);
      }
      else {
        frame += interval;
        continue;
      }
      camera._m00 *= -1.0; camera._m01 *= -1.0; camera._m02 *= -1.0;
      camera._m10 *= -1.0; camera._m11 *= -1.0; camera._m12 *= -1.0;
      camera._m20 *= -1.0; camera._m21 *= -1.0; camera._m22 *= -1.0;
    }
    ml::Matrix4x4<float> MVP = proj*camera.getInverse();
    MVP.transpose();
    glUniformMatrix4fv(depthMatrixID, 1, GL_FALSE, &MVP.matrix[0]);
    glShadeModel(GL_FLAT);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
    // Draw the triangle !
    glDrawElements(GL_TRIANGLES, indexvec.size(), GL_UNSIGNED_INT, (void*)0);
    glDisableVertexAttribArray(0);

    // Swap buffers
    DenseMatrixf ones(windowHeight, windowWidth, 0.0f);
    glfwSwapBuffers(window);
    {
        printf("Processing %d..\r", frame);
        fflush(stdout);
        cv::Mat1f tmp(windowHeight, windowWidth);
        glReadPixels(0, 0, windowWidth, windowHeight, GL_DEPTH_COMPONENT, GL_FLOAT, tmp.data);
        tmp = tmp * 2.0f - 1.0f;
        tmp = (2.0 * zNear * zFar) / (zFar + zNear - tmp * (zFar - zNear));
        cv::Mat1w out(windowHeight, windowWidth);
        cv::Mat1w maskedOut(windowHeight, windowWidth);
        maskedOut.setTo(0);
        tmp.convertTo(out, CV_16UC1, 1000.0);
        cv::Mat1b mask = out < ushort(zFar*1000-1);
        out.copyTo(maskedOut, mask);
        std::stringstream tmpFrame;
        tmpFrame << std::setw(4) << std::setfill('0') << frame;
        cv::imwrite((out_path + std::string("/frame-00") + tmpFrame.str() + ".depth.png").c_str(), maskedOut);
        frame += interval;
    }
  } // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

  printf("\n");
  // Cleanup VBO and shader
  glDeleteBuffers(1, &vertexbuffer);
  glDeleteBuffers(1, &indexbuffer);
  glDeleteProgram(depthProgramID);
  glDeleteVertexArrays(1, &VertexArrayID);

  // Close OpenGL window and terminate GLFW
  glfwTerminate();
}


int HeadlessRenderDepthFromMesh(MeshDataf &mesh, std::string poses_path, std::string out_path, int interval)
{
  GLFWwindow* window;

  // if (!glfwInit())
  // {
  //   fprintf(stderr, "Failed to initialize GLFW\n");
  //   getchar();
  //   return -1;
  // }
  int windowWidth = 640;
  int windowHeight = 480;

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Make invisible window since we render to FBO directly

  // // Open a window and create its OpenGL context
  // window = glfwCreateWindow(windowWidth, windowHeight, "Depth Renderer", NULL, NULL);
  // if (window == NULL){
  //   fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
  //   getchar();
  //   glfwTerminate();
  //   return -1;
  // }
  // glfwMakeContextCurrent(window);

  // Initialize GLEW
  glewExperimental = true; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glViewport(0, 0, windowWidth, windowHeight);

  // Dark blue background
  glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDisable(GL_MULTISAMPLE);

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);

  GLuint depthProgramID = LoadShaders("shaders/DepthRTT.vertexshader",
                  "shaders/DepthRTT.fragmentshader", "shaders/BerycentricGeometryShader.geometryshader");
  
  // Get a handle for our "MVP" uniform
  GLuint depthMatrixID = glGetUniformLocation(depthProgramID, "depthMVP");

  // Load it into a VBO
  GLuint vertexbuffer, indexbuffer;
  glGenBuffers(1, &vertexbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
  glBufferData(GL_ARRAY_BUFFER, mesh.m_Vertices.size()*sizeof(mesh.m_Vertices[0]), &mesh.m_Vertices[0], GL_STATIC_DRAW);

  std::vector<unsigned int> indexvec;
  for (int i = 0; i < mesh.m_FaceIndicesVertices.size(); i++)
    for (int j = 0; j < 3; j++)
      indexvec.push_back(mesh.m_FaceIndicesVertices[i][j]);
  glGenBuffers(1, &indexbuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexvec.size()*sizeof(unsigned int), &indexvec[0], GL_STATIC_DRAW);

  // zNear: 0.01, zFar: 30.0 (in meters)
  float zNear = 0.01;
  float zFar = 30.0;
  ml::Matrix4x4<float> camera, proj;
  {
    float tmp[] = {
      -1.81066, 0.00000, 0.00000, 0.00000,
      0.00000, -2.41421, 0.00000, 0.00000,
      0.00000, 0.00000, -(zFar + zNear) / (zFar - zNear), -2 * zFar*zNear / (zFar-zNear),
      0.00000, 0.00000, -1.00000, 0.00000
    };
    proj = ml::Matrix4x4<float>(tmp);
  }

  GLuint FramebufferName = 0;
  glGenFramebuffers(1, &FramebufferName);
  glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

  GLuint depthTexture;
  glGenTextures(1, &depthTexture);
  glBindTexture(GL_TEXTURE_2D, depthTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);
  // No color output in the bound framebuffer, only depth.
  glDrawBuffer(GL_NONE);

  // Always check that our framebuffer is ok
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    printf("something wrong in FramebufferStatus check\n");
    return false;
  }

  // Use our shader
  glUseProgram(depthProgramID);

  int frame = 0;
  do{
    // Clear the screen
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glViewport(0, 0, windowWidth, windowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
      std::stringstream tmpFrame;
      tmpFrame << std::setw(4) << std::setfill('0') << frame;
    //   FILE *in = fopen((std::string("") + target_scene + std::string("\\pose\\frame-00") + tmpFrame.str() + ".pose.txt").c_str(), "r");
      FILE *in = fopen((poses_path + std::string("/frame-00") + tmpFrame.str() + ".pose.txt").c_str(), "r");
      if (in == NULL)
        break;
      float tmp[4][4];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          fscanf(in, "%f", &tmp[i][j]);
      fclose(in);
      if (tmp[0][0] != -1.0) {
        camera = ml::Matrix4x4<float>((float*)&tmp[0][0]);
      }
      else {
        frame += interval;
        continue;
      }
      camera._m00 *= -1.0; camera._m01 *= -1.0; camera._m02 *= -1.0;
      camera._m10 *= -1.0; camera._m11 *= -1.0; camera._m12 *= -1.0;
      camera._m20 *= -1.0; camera._m21 *= -1.0; camera._m22 *= -1.0;
    }
    ml::Matrix4x4<float> MVP = proj*camera.getInverse();
    MVP.transpose();
    glUniformMatrix4fv(depthMatrixID, 1, GL_FALSE, &MVP.matrix[0]);
    glShadeModel(GL_FLAT);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
    // Draw the triangle !
    glDrawElements(GL_TRIANGLES, indexvec.size(), GL_UNSIGNED_INT, (void*)0);
    glDisableVertexAttribArray(0);

    // Swap buffers
    DenseMatrixf ones(windowHeight, windowWidth, 0.0f);
    glfwSwapBuffers(window);
    {
        printf("Processing %d..\r", frame);
        fflush(stdout);
        cv::Mat1f tmp(windowHeight, windowWidth);
        glReadPixels(0, 0, windowWidth, windowHeight, GL_DEPTH_COMPONENT, GL_FLOAT, tmp.data);
        tmp = tmp * 2.0f - 1.0f;
        tmp = (2.0 * zNear * zFar) / (zFar + zNear - tmp * (zFar - zNear));
        cv::Mat1w out(windowHeight, windowWidth);
        cv::Mat1w maskedOut(windowHeight, windowWidth);
        maskedOut.setTo(0);
        tmp.convertTo(out, CV_16UC1, 1000.0);
        cv::Mat1b mask = out < ushort(zFar*1000-1);
        out.copyTo(maskedOut, mask);
        std::stringstream tmpFrame;
        tmpFrame << std::setw(4) << std::setfill('0') << frame;
        cv::imwrite((out_path + std::string("/frame-00") + tmpFrame.str() + ".depth.png").c_str(), maskedOut);
        frame += interval;
    }
  } // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

  printf("\n");
  // Cleanup VBO and shader
  glDeleteBuffers(1, &vertexbuffer);
  glDeleteBuffers(1, &indexbuffer);
  glDeleteProgram(depthProgramID);
  glDeleteVertexArrays(1, &VertexArrayID);

  // Close OpenGL window and terminate GLFW
  glfwTerminate();
}

int main(int argc, char** argv)
{
    bool headless_render = true;
    if (argc < 5)
    {
      printf("Usage: ./depthRenderer <path_to_ply_model> <path_to_poses_files> <path_to_output_depth> <frame_interval>");
      exit(-1);
    }

    std::string mesh_path(argv[1]);
    std::string poses_path(argv[2]);
    std::string out_path(argv[3]);
    int interval = atoi(argv[4]);
    printf("Render '%s' to '%s' with interval %d\n", mesh_path.c_str(), out_path.c_str(), interval);
    MeshDataf mesh;
    MeshIOf::loadFromFile(mesh_path.c_str(), mesh);
    std::printf("Loaded a mesh with %d vertices\n", (int)mesh.m_Vertices.size());
    Timer t;
    
    if (headless_render)
    {
      // 1. Initialize EGL
      EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      EGLint major, minor;
      eglInitialize(eglDpy, &major, &minor);

      // 2. Select an appropriate configuration
      EGLint numConfigs;
      EGLConfig eglCfg;
      eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

      // 3. Create a surface
      EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

      // 4. Bind the API
      eglBindAPI(EGL_OPENGL_API);

      // 5. Create a context and make it current
      EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
      eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    
      // from now on use your OpenGL context
      HeadlessRenderDepthFromMesh(mesh, poses_path, out_path, interval);

      // 6. Terminate EGL when finished
      eglTerminate(eglDpy);
    }
    else
    {
      RenderDepthFromMesh(mesh, poses_path, out_path, interval);
    }
    cout << t.getElapsedTime() << "s" << endl;
	  return 0;
}
