import os, sys, glob
print("venv:", sys.prefix)
print("cublasLt DLL:", glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin", "cublasLt64_12.dll")))