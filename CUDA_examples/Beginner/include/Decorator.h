#include <iostream>
#include <time.h>

namespace Deco {

	template<class F, typename... Args>
	void TimeCheckDecorator(F func, Args ... args)
	{
		clock_t start, end;
		start = clock();
		func(args...);
		end = clock();

		//std::cout << "�� �ҿ� �ð� : " << end - start << std::endl;
		printf("�� �ҿ� �ð� : %4.6f \n",
			(double)((double)(end - start) / CLOCKS_PER_SEC));
	}
}
