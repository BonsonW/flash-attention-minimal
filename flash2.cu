#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int seqlen,
    const int nhead,
    const int headdim,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    const int win_l,
    const int win_r
) {
    int tx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int bz = blockIdx.z;
    int seq_stride = nhead * headdim;
    int tile_stride = nhead;

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (batch_idx * nhead * seqlen * headdim) + (head_idx * headdim);

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Br * headdim;
    float* Qi = sram;
    float* Oi = &sram[tile_size];
    float* Kj = &sram[tile_size * 2];
    float* Vj = &sram[tile_size * 2 + tile_size];
    float* S = &sram[tile_size * 2 + tile_size * 2];

    // Load Qi to SRAM
    for (int x = 0; x < headdim; x++) {
        Qi[(tx * headdim) + x] = Q[qkv_offset + (tile_size * bz * tile_stride) + (tx * seq_stride) + x];
        Oi[(tx * headdim) + x] = 0; // zero
    }
    
    float row_m_prev = -INFINITY;
    float row_l_prev = 0;
    float row_m_new, row_l_new;

    for (int j = 0; j < Tc; j++)  {
        // Load Kj, Vj to SRAM
        for (int x = 0; x < headdim; x++) {
            Kj[(tx * headdim) + x] = K[qkv_offset + (tile_size * j * tile_stride) + (tx * seq_stride) + x];
            Vj[(tx * headdim) + x] = V[qkv_offset + (tile_size * j * tile_stride) + (tx * seq_stride) + x];
        }
        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; y++) {
            float sum = 0;
            for (int x = 0; x < headdim; x++) {
                sum += Qi[(tx * headdim) + x] * Kj[(y * headdim) + x];
            }
            sum *= softmax_scale;

            bool out_winr = win_r >= 0 && (bz * Br) + tx < (j * Bc) + y - win_r;
            bool out_winl = win_l >= 0 && (bz * Br) + tx > (j * Bc) + y + win_l;
            S[(Bc * tx) + y] = out_winr || out_winl ? -INFINITY : sum;

            if (sum > row_m) row_m = sum;
        }
        
        // Compute new m
        row_m_new = max(row_m_prev, row_m);

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
            S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
            row_l += S[(Bc * tx) + y];
        }

        // Compute l
        row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

        // Write O, l, m to HBM
        for (int x = 0; x < headdim; x++) {
            float pv = 0;  // Pij * Vj
            for (int y = 0; y < Bc; y++) {
                pv += S[(Bc * tx) + y] * Vj[(y * headdim) + x];
            }
            Oi[(tx * headdim) + x] = (__expf(row_m_prev - row_m_new)) * Oi[(tx * headdim) + x] + pv;
        }

        // Update l, m
        row_l_prev = row_l_new;
        row_m_prev = row_m_new;
    }
    for (int x = 0; x < headdim; x++) {
        O[qkv_offset + (tile_size * bz * tile_stride) + (tx * seq_stride) + x] = 1 / row_l_new * Oi[(tx * headdim) + x];
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int win_l, int win_r) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;

    const int nbatch = Q.size(0); 
    const int seqlen = Q.size(1);
    const int nhead = Q.size(2);
    const int headdim = Q.size(3);

    const int Tc = ceil((float) seqlen / Bc);
    const int Tr = ceil((float) seqlen / Br);
    const float softmax_scale = 1.0 / sqrt(headdim);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Br * headdim * sizeof(float)) + (2 * Bc * headdim * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(nbatch, nhead, Tr);  // batch_size x num_heads x Tr
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        seqlen, nhead, headdim, Tc, Tr, Bc, Br, softmax_scale, win_l, win_r
    );
    return O;
}