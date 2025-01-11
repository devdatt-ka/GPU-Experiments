echo off
pushd ..\x64\debug\
CudaNPPProto.exe ..\..\data\sloth.png ..\..\data\sloth-boxfilter.png "box"
CudaNPPProto.exe ..\..\data\sloth.png ..\..\data\sloth-gaussfilter.png "gaussian"
CudaNPPProto.exe ..\..\data\sloth.png ..\..\data\sloth-sharpen.png "sharpen"
popd
echo on