Tested on fresh windows 10 install, GTX 1080:

- Install https://www.anaconda.com/products/individual 
  - filename Anaconda3-2020.07-Windows-x86_64.exe
  - Enable "Add Anaconda3 to my PATH environment variable" during installation

- Install cuda-10.2 
  - Download here : https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
  - filename cuda_10.2.89_441.22_win10.exe
  - Choose "Express installation" (it should upgrade drivers to 441.22)

- Install "Build Tools for Visual Studio 2019" https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
  - Enable 'C++ Build Tools' during installation

- add `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64` to PATH
  - If this directory doesn’t exist, you will need to determine where your `cl.exe` was installed. If this directory does exist add it to your PATH

- Reboot

- Install pycuda and pynacl
  - in cmd `pip install pycuda pynacl`

- download/extract lisk-cracker, open cmd inside lisk-cracker directory
- `python main.py --n-targets 20000` should run without errors
