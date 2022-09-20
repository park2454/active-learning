using MosekTools, Convex, LinearAlgebra, SparseArrays, FileIO, JLD2, Statistics, StatsBase, SCS, Random

logit(x) = log.(1 ./ (1 .- x))
logistic(x) = exp.(x) ./ ( 1 .+ exp.(x) )

function irls(X, y; μ=0.1)
    n,p = size(X)
    βold = βnew = zeros(p)
    while true
        mu = logistic(X*βold)
        w = mu .* (1 .- mu)
        βnew = βold + (X'Diagonal(w)*X + μ*I)\(X'*(y-mu)-μ*βold)
        if norm(βold-βnew) <= sqrt(eps(1.0))
            break
        end
        βold = copy(βnew)
    end
    return βnew
end

function sagnol_D(A, A0, nq; K=I, verbose=0, IC=0)
    if length(size(A)) == 1
        A = reshape(A,:,1)
    end
    n, m = size(A)
    
    if K==I
        K = Diagonal(ones(m))
    end
    
    k = size(K)[2]
    
    Z0 = Variable(size(A0)[1], k)
    if IC == 1
        select = Variable(n+1, :Bin)
    else
        select = Variable(n+1)
    end
    J = Variable(k, k)
    Z = Variable(n, k)
    T = Variable(n+1, k)
    
    if k > 1
        n_aux = 0
        N = k
        while N > 1
            N = ceil(N/2)
            n_aux += N
        end
        n_aux = Int(n_aux)
        aux = Variable(n_aux)
        problem = maximize(aux[end])
        N = k
        while N > 1 # geometric mean into socp
            if N%2 == 0
                N = Int(N/2)
                for i = 1:N
                    problem.constraints += norm([2*aux[i]; J[2i-1,2i-1]-J[2i,2i]], 2) <= J[2i-1,2i-1] + J[2i,2i]
                end
            else
                N = Int(ceil(N/2))
                for i = 1:N-1
                    problem.constraints += norm([2*aux[i]; J[2i-1,2i-1]-J[2i,2i]], 2) <= J[2i-1,2i-1] + J[2i,2i]
                end
                problem.constraints += norm([2*aux[N]; J[2N-1,2N-1]-1], 2) <= J[2N-1,2N-1]+1
            end
            pre = N
            prepre = 0
            while N > 1
                if N%2 == 0
                    N = Int(N/2)
                    for i = 1:N
                        problem.constraints += norm([2*aux[pre+i]; aux[prepre+2i-1]-aux[prepre+2i]], 2) <= aux[prepre+2i-1]+aux[prepre+2i]
                    end
                else
                    N = Int(ceil(N/2))
                    for i = 1:N-1
                        problem.constraints += norm([2*aux[pre+i]; aux[prepre+2i-1]-aux[prepre+2i]], 2) <= aux[prepre+2i-1]+aux[prepre+2i]
                    end
                    problem.constraints += norm([2*aux[pre+N]; aux[prepre+2N-1]-1], 2) <= aux[prepre+2N-1]+1
                end
                prepre = pre
                pre += N
            end
        end
        for i = 1:(k-1) # constraint 2
            problem.constraints += J[i, (i+1):k] == 0
        end
    else
        problem = maximize(J[1,1])
    end
    
    problem.constraints += A'*Z + A0'*Z0 == K*J # constraint 1
    
    for i = 1:n # constraint 3
        for j = 1:k
            problem.constraints += norm([2*Z[i,j]; T[i,j]-select[i]], 2) <= T[i,j]+select[i]
        end
    end
    
    for j = 1:k
        problem.constraints += norm([2*Z0[:,j]; T[n+1,j] - select[n+1]],2) <= T[n+1,j] + select[n+1] # constraint 3
        problem.constraints += sum(T[:,j]) <= J[j,j] # constraint 4
    end
    
    problem.constraints += T >= 0 # consraint 5
    
    problem.constraints += sum(select[1:n]) == nq # selection constraint
    problem.constraints += select[end] == 1
    
    if IC==0
        problem.constraints += select <= 1
    end
    
    solver = SCS.Optimizer(verbose = verbose)
    solve!(problem, solver)
    
    println(problem.status)
    return select.value[1:n]
end

function sagnol_A(A, A0, nq; K=I, verbose=0, IC=0)
    if length(size(A)) == 1
        A = reshape(A,:,1)
    end
    n, m = size(A)
    
    if K==I
        K = Diagonal(ones(m))
    end
    
    k = size(K)[2]
    
    Y0 = Variable(size(A0)[1], k)
    if IC == 1
        select = Variable(n+1, :Bin)
    else
        select = Variable(n+1)
    end
    Y = Variable(n,k)
    μ= Variable(n+1)
    
    problem = maximize(sum(μ))
    
    problem.constraints += A'*Y + A0'*Y0 == sum(μ)*K
    for i = 1:n
        problem.constraints += norm([2*vec(Y[i,:]); μ[i] - select[i]], 2) <= μ[i] + select[i]
    end
    problem.constraints += norm([2*vec(Y0); μ[n+1] - select[n+1]], 2) <= μ[n+1] + select[n+1]
    problem.constraints += μ >= 0 
    
    problem.constraints += select[n+1] == 1
    problem.constraints += sum(select[1:n]) == nq
    
    if IC==0
        problem.constraints += select <= 1
    end
    
    solver = SCS.Optimizer(verbose = verbose)
    solve!(problem, solver)
    
    println(problem.status)
    return select.value[1:n]
end

function auc(y,ŷ)
    CP = sum(y)
    CN = length(y) - CP
    cutoff = ones(length(y))*sort(unique([1; ŷ; 0]), rev=true)'
    pred = ŷ .> cutoff
    PP = vec(sum(pred, dims=1))
    TP = vec(sum((pred .+ y) .== 2, dims=1))
    FP = PP - TP
    TPR = [0; TP ./ CP; 1]
    FPR = [0; FP ./ CN; 1]
    H = 0.5 * (TPR[2:end] + TPR[1:end-1])
    W = diff(FPR)
    AUC = sum(H .* W)
    ACC = mean((ŷ .> 0.5) .== y)
    return AUC, ACC
end

function brute(X, k; μ=0.1)
    n = size(X)[1]
    cand = collect(combinations(1:n, k))
    val = zeros(length(cand))
    for i in 1:length(cand)
        X0 = X[cand[i],:]
        val[i] =  tr(X*((X0'X0+μ*I)\(X')))
    end
    return cand[findmin(val)[2]]
end

function brute2(X, k; μ=0.1)
    n = size(X)[1]
    cand = collect(combinations(1:n, k))
    best = 0
    val = Inf
    for i in 1:length(cand)
        X0 = X[cand[i],:]
        if tr(X*((X0'X0+μ*I)\(X'))) < val
            val = tr(X*((X0'X0+I)\(X')))
            best = i
        end
    end
    return cand[best]
end

function alg1(X, k; μ=0.1)
    K = X*X'
    n = size(X)[1]
    list = zeros(Int64, k)
    for i = 1:k
        val = zeros(n)
        for j = 1:n
            val[j] = sum(abs2, K[:,j]) / (K[j,j]+μ)
        end
        list[i] = findmax(val)[2]
        K -= 1/(K[list[i],list[i]]+μ)*K[:,list[i]]*K[:,list[i]]'
    end
    return list
end

function alg3(X,k;μ=0.1, γ=1)
    K = X*X'
    n = size(X)[1]

    α = zeros(n,n)
    β =  ones(n)
    label_old = label_new =  zeros(Int64, k) 
    cnt = 0
    
    while cnt < 10
        for i = 1:n
            α[:,i] = (Diagonal(1 ./ β) + K)\K[:,i]
        end

        for j = 1:n
            β[j] = sqrt(1/γ * sum(abs2, α[j,:]))
        end

        label_new = partialsortperm(β, 1:k, rev=true)
        if label_new == label_old
            cnt += 1
        else
            cnt = 0
        end
        label_old = copy(label_new)
    end
    
    return label_new
end

function alg4(X,k;μ=0.1, γ=1)
    K = X*X'
    n = size(X)[1]

    α = αnew = zeros(n,n)
    β = βnew =  ones(n)
    cnt = 0
    
    while cnt < 10
        for i = 1:n
            αnew[:,i] = (μ*Diagonal(1 ./ β) + K)\K[:,i]
        end

        for j = 1:n
            βnew[j] = sqrt(1/γ * sum(abs2, α[j,:]))
        end

        if norm(α-αnew) + norm(β-βnew) < 10^-13
            break
        end
        α=αnew
        β=βnew
    end
    
    return partialsortperm(βnew, 1:k, rev=true)
    #return norm(βnew), sort(sortperm(βnew)[1:k])
end