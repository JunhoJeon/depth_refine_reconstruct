# Depth renderer for ScanNet reconstructed models

You can use this code for rendering the ScanNet 3D model, to get the clean depth images for corresponding camera poses.

---
## Requirements

    opencv, mLib, glut, glew, glfw, EGL

You may already have GL and extensions on Ubuntu.

By default, [mLib](https://github.com/niessner/mLib) should be located at same level of renderer directory. 

You can change it by modifying MakeFlie.

### Desired folder hierachy

    this_repo/ --- egl_depthRenderer/ --- src/ -------- common/
                |- mLib/               |- Makefile   |- main.cpp
                                       |- ...        |- mLibSource.cpp
                                       |- README.md



---

## How to use it

The code needs reconstructed .ply model and camera poses from [ScanNet dataset](http://www.scan-net.org).

Camer pose files (frame-XXXXXX.pose.txt) contain 4x4 transformation matrix that project world to the camera coordinate system.

    $ cat frame-000000.pose.txt
    -0.955421 0.119616 -0.269932 2.65583
    0.295248 0.388339 -0.872939 2.9816
    0.000407581 -0.91372 -0.406343 1.36865
    0 0 0 1
    $ _

You can extract camera pose files of each scan from \*.sens file in ScanNet dataset (see [here](https://github.com/ScanNet/ScanNet/tree/master/SensReader)).

    $ make
    $ ./depthRenderer <path_to_ply_model> <path_to_poses_files> <path_to_output_depth> <frame_interval>
    
    For example:
    $ ./depthRenderer scene0000_00_vh_clean.ply poses depth_out 100
    Render 'scene0000_00_vh_clean.ply' to 'depth_out' with interval 100
    Loaded a mesh with 1990518 vertices
    Compiling shader : DepthRTT.vertexshader
    Compiling shader : DepthRTT.fragmentshader
    Compiling shader : BerycentricGeometryShader.geometryshader
    Linking program
    Processing 5500..
    0.333942s
    $ _

## Quality of rendered depth images
As you may know, rendered depth image can have different structure compared to originally captured raw depth image with same camera poses. It is caused by the reconstruction process may deform the 3D model during global error minimization, camera pose estimation errors, and so on.

To utilize the rendered clean depth for GT of supervised learning such as single-view depth estimation or depth refinement, several pre/post processing techniques can be options.

For example, before render the depth, you can locally re-adjust the camera pose using ICP or similar methods to achieve better alignment between rendered clean depth image and raw depth image (and color image). Or, you can just crop the local patches whose structure similarity to the raw depth patch is above the threshold to build a patch-wise dataset (like our ECCV paper).

## FAQ
- Does it only work with ScanNet dataset?
  - With a little modification, you can use this code to render any 3D models (meshes) with arbitrary camera poses and parameters.
- Why it took so long time to publish this code?
  - Preparing graduation.. and first renderer was written in Unity scripts, but Unity cannot handle well such a huge 3D model with millions of vertices. So I re-implemented the depth renderer.

## TODO
- ~~Support headless rendering through SSH without X-window~~ (Done)
- discard opencv dependency (may use stb_image)
- support direct pose extraction from *.sens files.

## Contributors
Junho Jeon (zwitterion27@gmail.com)

Jinwoong Jung (jinwoong.jung@postech.ac.kr)
