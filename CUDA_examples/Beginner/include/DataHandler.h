#include <time.h>
#include <stdio.h>
#include <iostream>

namespace D_Handler
{
	template<typename T>
	void initialize_1d_array(T* src, int size)
	{
		time_t t;
		srand((unsigned)time(&t));

		for (int i = 0; i < size; ++i)
		{
			src[i] = (T)(rand() & 0xFF);
		}
	}

	template<typename T>
	void initialize_2d_array(T* src, int nx, int ny)
	{
		time_t t;
		srand((unsigned)time(&t));

		for (int y = 0; y < ny; ++y)
		{
			for (int x = 0; x < nx; ++x)
			{
				src[y][x] = (T)(rand() & 0xFF);
			}
		}
	}

	template<typename T>
	void compare_1d_array(T* src, T* dst, int size)
	{
		for (int i = 0; i < size; ++i)
		{
			if (src[i] != dst[i])
			{
				printf("Arrays are not Equal\n");
				printf("%d / %d", src[i], dst[i]);
			}
		}
		printf("Arrays are Equal!\n");
	}
	template<typename T>
	void compare_2d_array(T* src, T* dst, int nx, int ny)
	{
		for (int y = 0; y < ny; ++y)
		{
			for (int x = 0; x < nx; ++x)
			{
				if (src[y][x] != dst[y][x])
				{
					printf("Arrays are not Equal\n");
					printf("%d / %d", src[y][x], dst[y][x]);
				}
			}
		}
		printf("Arrays are Equal!\n");
	}

	template<typename T>
	void cpu_sum_array(T* input1, T* input2, T* out, int size)
	{
		for (int idx = 0; idx < size; ++idx)
		{
			out[idx] = input1[idx] + input2[idx];
		}
	}
	template<typename T>
	void cpu_sum_2d_array(T* input1, T*input2, T* out, int nx, int ny)
	{
		int size = nx * ny;
		for (int y = 0; y < size; ++y)
		{
			for (int x = 0; x < size; ++x)
			{
				out[y][x] = input1[y][x] + input2[y][x];
			}
		}
	}
		
}
