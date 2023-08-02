#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernUpSweep(int n, int d, int *dev_data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
                return;
            int offset = 1 << (d + 1);
            if ((index + 1) % offset == 0)
            {
                int prev = index - (1 << d);
                dev_data[index] += dev_data[prev];
            }
        }

        __global__ void kernDownSweep(int n, int d, int *dev_data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
                return;
            int offset = 1 << (d + 1);
            if ((index + 1) % offset == 0)
            {
                int prev = index - (1 << d);
                int temp = dev_data[prev];
                dev_data[prev] = dev_data[index];
                dev_data[index] += temp;
            }
        }

        __global__ void setIndexAsZero(int n, int *odata)
        {
            odata[n] = 0;
        }

        void scan(int n, int *odata, const int *idata)
        {
            // TODO
            int blockSize = 256;
            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);

            int num_levels = ilog2ceil(n);
            int cudaArraySize = 1 << num_levels;
            int *dev_data;

            cudaMalloc((void **)&dev_data, cudaArraySize * sizeof(int));
            cudaMemset(dev_data, 0, n * sizeof(int));                             // initialize to 0
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy data to device

            timer().startGpuTimer();
            for (int d = 0; d < num_levels; d++)
            {
                // std::cout << "up sweep d: " << d << std::endl;
                // build sum in place up the tree
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(cudaArraySize, d, dev_data);
                // int *debug = new int[n];
                // cudaMemcpy(debug, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
                // for (int i = 0; i < n; i++)
                // {
                //     std::cout << debug[i] << " ";
                // }
                // std::cout << std::endl;

                cudaDeviceSynchronize();
            }
            setIndexAsZero<<<1, 1>>>(cudaArraySize - 1, dev_data);
            for (int d = num_levels - 1; d >= 0; d--)
            {
                // std::cout << "down sweep d: " << d << std::endl;
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(cudaArraySize, d, dev_data);
                // int *debug = new int[n];
                // cudaMemcpy(debug, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
                // for (int i = 0; i < n; i++)
                // {
                //     std::cout << debug[i] << " ";
                // }
                // std::cout << std::endl;

                cudaDeviceSynchronize();
            }
            // TODO
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata)
        {
            // TODO

            // scan - initialize
            int blockSize = 256;
            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);
            int num_levels = ilog2ceil(n);
            int cudaArraySize = 1 << num_levels;
            int count = 0;
            int *dev_count;
            int *dev_data;    // holder for scan result
            int *dev_bools;   // holder for boolen array
            int *dev_inputs;  // copy of idata
            int *dev_outputs; // copy of odata
            cudaMalloc((void **)&dev_count, sizeof(int));
            cudaMalloc((void **)&dev_data, cudaArraySize * sizeof(int));
            cudaMalloc((void **)&dev_bools, cudaArraySize * sizeof(int));
            cudaMalloc((void **)&dev_inputs, n * sizeof(int));
            cudaMalloc((void **)&dev_outputs, n * sizeof(int));
            cudaMemset(dev_count, 0, sizeof(int));                                   // initialize to 0, size is 2^ilog2ceil(n
            cudaMemset(dev_data, 0, cudaArraySize * sizeof(int));                    // initialize to 0, size is 2^ilog2ceil(n)
            cudaMemset(dev_bools, 0, cudaArraySize * sizeof(int));                   // initialize to 0, size is 2^ilog2ceil(n)
            cudaMemcpy(dev_inputs, idata, n * sizeof(int), cudaMemcpyHostToDevice);  // copy data to device
            cudaMemcpy(dev_outputs, odata, n * sizeof(int), cudaMemcpyHostToDevice); // copy data to device

            // scan - prepare boolen array, get count
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_inputs, dev_count);

            cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(dev_data, dev_bools, cudaArraySize * sizeof(int), cudaMemcpyDeviceToDevice);

            // scan - scan boolean array
            timer().startGpuTimer();
            for (int d = 0; d < num_levels; d++)
            {
                // build sum in place up the tree
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(cudaArraySize, d, dev_data);
                cudaDeviceSynchronize();
            }
            setIndexAsZero<<<1, 1>>>(cudaArraySize - 1, dev_data);
            for (int d = num_levels - 1; d >= 0; d--)
            {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(cudaArraySize, d, dev_data);

                cudaDeviceSynchronize();
            }
            // scan finished, next scatter
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid,
                                                    blockSize>>>(n, dev_outputs, dev_inputs, dev_bools, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_outputs, count * sizeof(int), cudaMemcpyDeviceToHost);

            return count;
        }
    }
}
