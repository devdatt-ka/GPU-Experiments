# GPU-Experiments
This is repository for experimental code on CUDA.

#Overview

The Project CudaNPPProto is an example code for usage of CUDA NPP library. Specifically it demonstrates usage of different filters provided by the NPP.
The project is created using Microsoft Visual Studio 2022 and follow the Visual Studio sturucture and build process. The project uses opencv libraies to read and write images.

## Code Organization

```CudaNPPProto/```
This folder contains solution and Visual studio project file.

```CudaNPPProto/source/```
This folder contains C++ source file and header file. It also has the batch file that runs to create different output format images using differtent filter options. 

```CudaNPPProto/data/```
This folder has "sloth.png" file used for testing the code. 

```CudaNPPProto/X64/```
This folder is not part of the repository but will be created when you build the solution. It will contain 'debug' and 'release' folders where the executable CudaNPPProto.exe will be created. The executable takes three argumants - path of the input image, path where output image will be written, and a string describing the filter option. Currently supported filters are "box", "gaussian", and "sharpen". 

``` CudaNPPProto.exe <path of input image> <path of output image> "box" ```