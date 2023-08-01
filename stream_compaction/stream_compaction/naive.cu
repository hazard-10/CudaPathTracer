#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        __global__ void setFirstAsZero(int *odata)
        {
            odata[0] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scan(int n, int *odata, const int *idata)
        {
            int blockSize = 128;
            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);

            // TODO
            int d_max = ilog2ceil(n);
            int *dstFirst;
            int *dstSecond;
            cudaMalloc((void **)&dstFirst, n * sizeof(int));
            cudaMalloc((void **)&dstSecond, n * sizeof(int));

            setFirstAsZero<<<1, 1>>>(dstFirst);
            cudaMemcpy(&dstFirst[1], idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int i = 1; i <= d_max; i++)
            {
                int blockSize = n;
                int d_offset = 1 << (i - 1);
                kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, d_offset, dstSecond, dstFirst);
                std::swap(dstFirst, dstSecond);
            }
            timer().endGpuTimer();
            // setFirstAsZero<<<1, 1>>>(dstFirst);
            cudaMemcpy(odata, dstFirst, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dstFirst);
            cudaFree(dstSecond);
        }
    }
}
