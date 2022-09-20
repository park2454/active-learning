using Random, Combinatorics, LinearAlgebra, Convex, Mosek, BenchmarkTools, DelimitedFiles, StatsBase, JLD2, FileIO
include("functions.jl")

Random.seed!(1992)

μ = 0.1
γ = 1
k = collect(5:5:50)

mse = zeros(10,5,100)
mse_test = zeros(10,5,100)

for rep = 1:100
    wifi = readdlm("dataset/trainingData.csv",',',Any,'\n')
    colnames = wifi[1,:]
    wifi = wifi[2:end,1:end-1]
    n, p = size(wifi)
    wifi = wifi[sample(1:n, 100, replace = false),:]
    y = convert(Array{Float64,1}, wifi[:,523])
    X = convert(Array{Float64, 2}, wifi[:,1:200])
    n, p = size(X)

    test = readdlm("dataset/validationData.csv",',',Any,'\n')
    test = test[2:end,:]
    n_test, p_test = size(test)
    test = test[sample(1:n_test, 100, replace = false),:]
    y_test = convert(Array{Float64,1}, test[:,523])
    X_test = convert(Array{Float64, 2}, test[:,1:200])

    for i = 1:length(k)
        # SOCP - sampling w/ probability
        socp = sagnol_A(X, μ*Diagonal(ones(p)), k[i]; K=X', verbose=0, IC=0)
        cand = (1:n)[rand(n) .< socp]
        X0 = X[cand,:]
        y0 = y[cand]
        β = (X0'X0+μ*I)\(X0'y0)
        mse[i,1,rep] = sum(abs2, y - X*β) / length(y)
        mse_test[i,1,rep] = sum(abs2, y_test - X_test*β) / length(y_test)
    
        # SOCP - top k candidate
        cand = partialsortperm(socp, 1:k[i], rev=true)
        X0 = X[cand,:]
        y0 = y[cand]
        β = (X0'X0+μ*I)\(X0'y0)
        mse[i,2,rep] = sum(abs2, y - X*β) / length(y)
        mse_test[i,2,rep] = sum(abs2, y_test - X_test*β) / length(y_test)

        # sequential
        cand = alg1(X,k[i])
        X0 = X[cand,:]
        y0 = y[cand]
        β = (X0'X0+μ*I)\(X0'y0)
        mse[i,3,rep] = sum(abs2, y - X*β) / length(y)
        mse_test[i,3,rep] = sum(abs2, y_test - X_test*β) / length(y_test)

        # relaxation
        cand = alg4(X,k[i])
        X0 = X[cand,:]
        y0 = y[cand]
        β = (X0'X0+μ*I)\(X0'y0)
        mse[i,4,rep] = sum(abs2, y - X*β) / length(y)
        mse_test[i,4,rep] = sum(abs2, y_test - X_test*β) / length(y_test)
    
        # random sampling
        cand = sample(1:n, k[i] ,replace=false)
        X0 = X[cand,:]
        y0 = y[cand]
        β = (X0'X0+μ*I)\(X0'y0)
        mse[i,5,rep] = sum(abs2, y - X*β) / length(y)
        mse_test[i,5,rep] = sum(abs2, y_test - X_test*β) / length(y_test)

        @save "comparison.jld2" mse mse_test
        println("$i / 10 iteration in $rep / 100 repitition complete")
    end
end