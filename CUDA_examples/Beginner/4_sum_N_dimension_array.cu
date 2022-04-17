/*
<description>
- 다차원 데이터를 핸들링 한다.
*/
#include "include/common.cuh"
#include "include/DataHandler.h"
#include "include/Decorator.h"

#include "cuda_runtime.h"
#include "cuda.h"

#include <stdio.h>
#include <iostream>

__global__ void sum_1D_grid_1D_block(float* in1, float* in2, float* out, int nx)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out[tid] = in1[tid] + in2[tid];
}

__global__ void sum_2D_grid_2D_block(float* in1, float* in2, float* out, int nx, int ny)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = nx * tidy + tidx;

	if (tidx < nx && tidy < ny)
		out[tid] = in1[tid] + in2[tid];
}

void sum_1d_array_test()
{
	printf("===== 1D GRID, BLOCK TEST =====\n");
	int size = 1 << 10;
	int block_size = 128;

	// 1D PARAMETER
	dim3 block1(block_size);
	int grid_size = (size + block1.x - 1) / block1.x;
	dim3 grid1(grid_size);
	printf("SIZE : %d  /  GRID * BLOCK : %d\n", size, block_size * grid_size);

	float* in1, * in2, * cpu_out, * gpu_out;
	unsigned int byte_size = size * sizeof(float);

	in1 = (float*)malloc(byte_size);
	in2 = (float*)malloc(byte_size);
	cpu_out = (float*)malloc(byte_size);
	gpu_out = (float*)malloc(byte_size);

	D_Handler::initialize_1d_array<float>(in1, size);
	D_Handler::initialize_1d_array<float>(in2, size);
	D_Handler::cpu_sum_array(in1, in2, cpu_out, size);

	// 1D GPU
	float* d_in1, * d_in2, * d_out;
	GPU_ERROR_CHECK(cudaMalloc((void**)&d_in1, byte_size));
	GPU_ERROR_CHECK(cudaMalloc((void**)&d_in2, byte_size));
	GPU_ERROR_CHECK(cudaMalloc((void**)&d_out, byte_size));
	GPU_ERROR_CHECK(cudaMemset(d_out, 0, byte_size));

	GPU_ERROR_CHECK(cudaMemcpy(d_in1, in1, byte_size, cudaMemcpyHostToDevice));
	GPU_ERROR_CHECK(cudaMemcpy(d_in2, in2, byte_size, cudaMemcpyHostToDevice));

	sum_1D_grid_1D_block << <grid1, block1 >> > (d_in1, d_in2, d_out, size);

	GPU_ERROR_CHECK(cudaDeviceSynchronize());
	GPU_ERROR_CHECK(cudaMemcpy(gpu_out, d_out, byte_size, cudaMemcpyDeviceToHost));

	D_Handler::compare_1d_array(cpu_out, gpu_out, size);

	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	free(in1);
	free(in2);
	free(cpu_out);
	free(gpu_out);
}

void sum_2d_array_test()
{
	printf("===== 2D GRID, BLOCK TEST =====\n");
	int size = 1 << 22;
	int nx = 1 << 14;
	int ny = size / nx;

	// 알아서 커스텀
	int block_x = 128;
	int block_y = 8;
	int byte_size = size * sizeof(float);

	dim3 block(block_x, block_y);
	dim3 grid((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y);
	
	printf("nx : %d , ny : %d\n", nx, ny);

	// HOST 메모리 할당
	float* host_in1, * host_in2, * cpu_out, * gpu_out;
	host_in1 = (float*)malloc(byte_size);
	host_in2 = (float*)malloc(byte_size);
	cpu_out = (float*)malloc(byte_size);
	gpu_out = (float*)malloc(byte_size);

	D_Handler::initialize_1d_array<float>(host_in1, size);
	D_Handler::initialize_1d_array<float>(host_in2, size);
	D_Handler::cpu_sum_array<float>(host_in1, host_in2, cpu_out, size);

	// GPU 메모리 할당
	float* device_in1, * device_in2, * device_out;
	GPU_ERROR_CHECK(cudaMalloc((void**)&device_in1, byte_size));
	GPU_ERROR_CHECK(cudaMalloc((void**)&device_in2, byte_size));
	GPU_ERROR_CHECK(cudaMalloc((void**)&device_out, byte_size));

	GPU_ERROR_CHECK(cudaMemcpy(device_in1, host_in1, byte_size, cudaMemcpyHostToDevice));
	GPU_ERROR_CHECK(cudaMemcpy(device_in2, host_in2, byte_size, cudaMemcpyHostToDevice));
	sum_2D_grid_2D_block << <grid, block >> > (device_in1, device_in2, device_out, nx, ny);
	
	GPU_ERROR_CHECK(cudaDeviceSynchronize());
	GPU_ERROR_CHECK(cudaMemcpy(gpu_out, device_out, byte_size, cudaMemcpyDeviceToHost));
	
	D_Handler::compare_1d_array(gpu_out, cpu_out, size);

	GPU_ERROR_CHECK(cudaFree(device_in1));
	GPU_ERROR_CHECK(cudaFree(device_in2));
	GPU_ERROR_CHECK(cudaFree(device_out));
	free(gpu_out);
	free(cpu_out);
	free(host_in1);
	free(host_in2);
}

int main()
{
	sum_1d_array_test();
	sum_2d_array_test();

	return 0;
}