#pragma once

#include <stdexcept>
#include <sstream>
#include <cuda_runtime_api.h>

namespace cuda
{
	template<typename F, typename N>
	void check_error(const ::cudaError_t e, F&& f, N&& n)
	{
		if(e != ::cudaSuccess)
		{
			std::stringstream s;
			s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
			throw std::runtime_error{s.str()};
		}
	}
}
#define CHECK_CUDA_ERROR(e) (cuda::check_error(e, __FILE__, __LINE__))
