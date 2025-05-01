#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* q, const float* k, const float* v, const int seqlen, const int nheads, const int headdim,
                    const int Tc, const int Tr, const int batch_cols, const int batch_rows, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_stride = nheads * headdim;

    // offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (batch_idx * seqlen * nheads * headdim) + (head_idx * headdim);
    int lm_offset = (batch_idx * nheads * seqlen) + (head_idx * seqlen); // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = batch_cols * headdim;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < headdim; x++) {
            Kj[(tx * headdim) + x] = k[qkv_offset + (tile_size * j) + (tx * seq_stride) + x];
            Vj[(tx * headdim) + x] = v[qkv_offset + (tile_size * j) + (tx * seq_stride) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < headdim; x++) {
                Qi[(tx * headdim) + x] = q[qkv_offset + (tile_size * i) + (tx * seq_stride) + x];
            }
            float row_m_prev = m[lm_offset + (batch_rows * i) + tx];
            float row_l_prev = l[lm_offset + (batch_rows * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < batch_cols; y++) {
                float sum = 0;
                for (int x = 0; x < headdim; x++) {
                    sum += Qi[(tx * headdim) + x] * Kj[(y * headdim) + x];
                }
                sum *= softmax_scale;
                S[(batch_cols * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < batch_cols; y++) {
                S[(batch_cols * tx) + y] = __expf(S[(batch_cols * tx) + y] - row_m);
                row_l += S[(batch_cols * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < headdim; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < batch_cols; y++) {
                    pv += S[(batch_cols * tx) + y] * Vj[(y * headdim) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * seq_stride) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * seq_stride) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (batch_rows * i) + tx] = row_m_new;
            l[lm_offset + (batch_rows * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    // TODO: determine Bc, Br dynamically
    const int batch_cols = 32;
    const int batch_rows = 32;

    const int nbatch = q.size(0);
    const int seqlen = q.size(1);
    const int nheads = q.size(2);
    const int headdim = q.size(3);

    const int Tc = ceil((float) seqlen / batch_cols);
    const int Tr = ceil((float) seqlen / batch_rows);
    const float softmax_scale = 1.0 / sqrt(headdim);

    // Initialize O, l, m to HBM
    auto out = torch::zeros_like(q);
    auto l = torch::zeros({nbatch, nheads, seqlen});
    auto m = torch::full({nbatch, nheads, seqlen}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * batch_cols * headdim * sizeof(float)) + (batch_cols * batch_rows * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(nbatch, nheads);  // batch_size x num_heads
    dim3 block_dim(batch_cols);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        seqlen, nheads, headdim, Tc, Tr, batch_cols, batch_rows, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), out.data_ptr<float>()
    );
    return out;
}