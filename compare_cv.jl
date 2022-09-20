using Random, Combinatorics, LinearAlgebra, Convex, Mosek, BenchmarkTools, DelimitedFiles, StatsBase, JLD2, FileIO
include("functions.jl")

Random.seed!(1992)

μ = 0.1
γ = 1
k = collect(5:5:50)

acc_train = zeros(10,7,100)
auc_train = zeros(10,7,100)

acc_test = zeros(10,7,100)
auc_test = zeros(10,7,100)

for rep = 1:100
    sonar = readdlm("dataset/sonar.all-data",',',Any,'\n')
    y_sonar = sonar[:,end]
    sonar = convert(Array{Float64,2}, sonar[:,1:end-1])
    y_sonar[findall(x->x=="M",y_sonar)] .= 1
    y_sonar[findall(x->x=="R",y_sonar)] .= 0
    y_sonar = convert(Array{Float64, 1}, y_sonar)
    n, p = size(sonar)

    test = sample(1:n, 20, replace=false)
    X_test = sonar[test,:]
    y_test = y_sonar[test]
    
    train = setdiff(1:n,test)
    X = sonar[train,:]
    y = y_sonar[train]
    n, p = size(X)

    for i = 1:length(k)
        # OED, SOCP - sampling w/ probability
        socp = sagnol_A(X, μ*Diagonal(ones(p)), k[i]; verbose=0, IC=0)
        cand = (1:n)[rand(n) .< socp]
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,1,rep], acc_train[i,1,rep] = auc(y, logistic.(X*β))
        auc_test[i,1,rep], acc_test[i,1,rep] = auc(y_test, logistic.(X_test*β))
    
        # OED, SOCP - top k candidate
        cand = partialsortperm(socp, 1:k[i], rev=true)
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,2,rep], acc_train[i,2,rep] = auc(y, logistic.(X*β))
        auc_test[i,2,rep], acc_test[i,2,rep] = auc(y_test, logistic.(X_test*β))
        
        # TED, SOCP - sampling w/ probability
        socp = sagnol_A(X, μ*Diagonal(ones(p)), k[i]; K=X', verbose=0, IC=0)
        cand = (1:n)[rand(n) .< socp]
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,3,rep], acc_train[i,3,rep] = auc(y, logistic.(X*β))
        auc_test[i,3,rep], acc_test[i,3,rep] = auc(y_test, logistic.(X_test*β))
    
        # TED, SOCP - top k candidate
        cand = partialsortperm(socp, 1:k[i], rev=true)
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,4,rep], acc_train[i,4,rep] = auc(y, logistic.(X*β))
        auc_test[i,4,rep], acc_test[i,4,rep] = auc(y_test, logistic.(X_test*β))

        # sequential
        cand = alg1(X,k[i])
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,5,rep], acc_train[i,5,rep] = auc(y, logistic.(X*β))
        auc_test[i,5,rep], acc_test[i,5,rep] = auc(y_test, logistic.(X_test*β))

        # relaxation
        cand = alg4(X,k[i])
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,6,rep], acc_train[i,6,rep] = auc(y, logistic.(X*β))
        auc_test[i,6,rep], acc_test[i,6,rep] = auc(y_test, logistic.(X_test*β))
    
        # random sampling
        cand = sample(1:n, k[i] ,replace=false)
        β = irls(X[cand,:],y[cand],μ=μ)
        auc_train[i,7,rep], acc_train[i,7,rep] = auc(y, logistic.(X*β))
        auc_test[i,7,rep], acc_test[i,7,rep] = auc(y_test, logistic.(X_test*β))

        @save "comparison2.jld2" acc_train auc_train acc_test auc_test
        println("$i / 10 iteration in $rep / 100 repitition complete")
    end
end