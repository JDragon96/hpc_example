/*
<compile>
$ 

<description>
input1 + input2 => out
*/
#include "include/common.cuh"
#include "include/DataHandler.h"
#include "include/Decorator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>

__global__ void gpu_sum_array(int* input1, int* input2, int* out, int size)
{
	// 1D Array에 대한 thread Index 계산
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// size => output array의 길이
	if (idx < size)
		out[idx] = input1[idx] + input2[idx];
}

//
//int main()
//{
//	// PARAMETER
//	int size = 1 << 22;
//	int block_size = 256;
//	int num_of_bytes = sizeof(int) * size;
//
//	// HOST Variables
//	// HOST = GPU, DEVICE = CPU
//	int *host_input1, *host_input2, *host_out, *device_out;
//
//	host_input1 = (int*)malloc(num_of_bytes);
//	host_input2 = (int*)malloc(num_of_bytes);
//	host_out = (int*)malloc(num_of_bytes);
//	device_out = (int*)malloc(num_of_bytes);
//
//	D_Handler::initialize_1d_array(host_input1, size);
//	D_Handler::initialize_1d_array(host_input2, size);
//
//	memset(host_out, 0, num_of_bytes);
//	memset(device_out, 0, num_of_bytes);
//
//	// CPU TEST
//	Deco::TimeCheckDecorator(cpu_sum_array, host_input1, host_input2, host_out, size);
//
//	// DEVICE Variables
//	int* device_input1, *device_input2, *device_result;
//	GPU_ERROR_CHECK(cudaMalloc((int**)&device_input1, num_of_bytes));
//	GPU_ERROR_CHECK(cudaMalloc((int**)&device_input2, num_of_bytes));
//	GPU_ERROR_CHECK(cudaMalloc((int**)&device_result, num_of_bytes));
//
//	dim3 block(block_size);
//	dim3 grid((size / block.x) + 1);
//
//	GPU_ERROR_CHECK(cudaMemcpy(device_input1, host_input1, num_of_bytes, cudaMemcpyHostToDevice));
//	GPU_ERROR_CHECK(cudaMemcpy(device_input2, host_input2, num_of_bytes, cudaMemcpyHostToDevice));
//
//	// GPU 연산 시작
//	clock_t start, end;
//	start = clock();
//	gpu_sum_array<<<grid, block>>>(device_input1, device_input2, device_result, size);
//	end = clock();
//	printf("총 소요 시간 : %4.6f \n",
//		(double)((double)(end - start) / CLOCKS_PER_SEC));
//	
//	// GPU SYNCHRONIZE
//	GPU_ERROR_CHECK(cudaDeviceSynchronize());
//
//	// Device to Host
//	GPU_ERROR_CHECK(cudaMemcpy(device_out, device_result, num_of_bytes, cudaMemcpyDeviceToHost));
//
//	// Compare results
//	D_Handler::compare_1d_array<int>(host_out, device_out, size);
//
//	// Memory Free
//	GPU_ERROR_CHECK(cudaFree(device_input1));
//	GPU_ERROR_CHECK(cudaFree(device_input2));
//	GPU_ERROR_CHECK(cudaFree(device_result));
//	free(host_input1);
//	free(host_input2);
//	free(host_out);
//	free(device_out);
//
//	return 0;
//}