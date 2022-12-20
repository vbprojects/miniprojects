function geninv(K::Matrix{Float64})
    m, n = size(K)
    A = K'K
    dA = diag(A)
    tol = minimum(dA.>0)*1e-9
    L = zeros(size(A))
    r = 0
    for k in 1:n
        r = r+1
        L[k:n, r] = A[k:n, k] .- L[k:n, 1:(r-1)] * (L[k, 1:(r-1)]')
        if L[k, r] > tol
            L[k, r] = sqrt(L[k, r])
            if k < n
                L[(k+1):n, r] = L[(k+1):n, r]/L[k, r]
            end
        else
            r = r-1
        end
    end
end

X = 1:.1:10 |> collect
rbf(x) = (x .- X) .^ 2 .|> mult(-1) .|> div(2 * 1.0 ^ 2) .|> exp
K = reduce(hcat, rbf.(X))
geninv(K)
m, n = size(K)
A = K'K
dA = diag(A)
tol = minimum(dA.>0)*1e-9
L = zeros(size(A))
r = 0
for k in 1:n
    # k = 2
    r = r+1
    # r = 2
    if r == 1
        L[k:n, r] = A[k:n, k]
    else
        L[k:n, r] = A[k:n, k] .- L[k:n, 1:(r-1)] * (L[k, 1:(r-1)]')
    end
    L[k:n, r] = A[k:n, k] .- L[k:n, 1:(r-1)] * (L[k, 1:(r-1)]')
    if L[k, r] > tol
        L[k, r] = sqrt(L[k, r])
        if k < n
            L[(k+1):n, r] = L[(k+1):n, r]/L[k, r]
        end
    else
        r = r-1
    end
end