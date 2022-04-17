/*
<description>
about warp divergence
- divergence�� �����ս� ����� ���� ���
- ������ warp�� �����ϴ� thread���� �ٸ� ����� �����ؾ� ���ļ��� ������
*/
#include "include/common.cuh"
#include "include/DataHandler.h"
#include "include/Decorator.h"

#include "cuda_runtime.h"
#include "cuda.h"

#include <stdio.h>
#include <iostream>

__global__ void without_divergence()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Warp�� thread ������ 32��	
	// GPU �����ͽ�Ʈ Ȯ��
	int warp_id = tid / 32;

	float a, b;
	a = b = 0;

	// ������ warp�� �����ϴ� ������� ������ ����� �����ϵ���
	if (warp_id % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200;
		b = 75;
	}
}

__global__ void divergence()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = tid / 32;

	float a, b;
	a = b = 0;

	// �������� �����尡 �ٸ� �۾��� �����ϵ���
	if (tid % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200;
		b = 75;
	}
}


//int main()
//{
//	printf("=== WARP DIVERGENCE EXAMPLE ===\n");
//
//	// PARAMETER SETTING
//	int size = 1 << 22;
//	int block_size = 128;
//	dim3 block(block_size);
//	int grid_size = (size + block.x - 1) / block.x;
//
//	dim3 grid(grid_size);
//
//	printf("BLOCK : %d\n", block_size);
//	printf("GRID : %d\n", grid_size);
//	printf("SIZE : %d\n", size);
//	printf("GRID * BLOCK : %d\n", grid_size * block_size);
//
//	// GPU
//	clock_t start, end;
//	start = clock();
//	without_divergence << <grid, block >> > ();
//	GPU_ERROR_CHECK(cudaDeviceSynchronize());
//	end = clock();
//	printf("NOT DIVERGENCE : %4.6f\n", end - start);
//
//	start = clock();
//	divergence << <grid, block >> > ();
//	GPU_ERROR_CHECK(cudaDeviceSynchronize());
//	end = clock();
//	printf("DIVERGENCE : %4.6f\n", end - start);
//
//	return 0;
//}