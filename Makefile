all:
	nvcc a4.cu -D_FILE_OFFSET_BITS=64 -Xcompiler -fopenmp -o a4