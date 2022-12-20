using LambdaFn, FLoops, LinearAlgebra, StatsBase, Random, Distributions, Plots, CUDA

using CUDA
using ImageFiltering
using NNlibCUDA
using KernelAbstractions

I = rand([1, 0], 100, 100)
K = [1 1 1; 1 0 1; 1 1 1]

function cconv(I, K)
    M, N = size(I)
    # nI = zeros(M, N)
    P, Q = size(K)
    nI = zeros(M, N)
    c = collect
    pc = (P + 1) / 2 |> Int
    qc = (Q + 1) / 2 |> Int
    
    @floop for i in (1):(M) |> c
        for j in (1):(N) |> c
            for p in 1:P |> c
                @inbounds for q in 1:Q |> c
                    ni = i + p - pc
                    ni = ni + (ni > M) * -(M-1) + (ni < 1) * (M)
                    nj = j + q - qc
                    nj = nj + (nj > N) * -(N-1) + (nj < 1) * (N)
                    nI[i, j] += K[p, q] * I[ni, nj]
                end
            end
        end
    end
    return nI
end


cconv(I, K)


(3+1)/2 |> Int
1:3 |> collect

-1 + 