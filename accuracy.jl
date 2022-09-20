start = 67

include("functions.jl")
using GLM, DataFrames, Statistics, DelimitedFiles, Random, Dates, LinearAlgebra
import StatsBase.sample

n = 250
magic = readdlm("magic04.data", ',', Any, '\n')
magic = magic[sample(1:19020, n, replace = false),:]
loc = findall(x->x=="g",magic[:,11])
magic[loc,11] .= 1
loc = findall(x->x=="h",magic[:,11])
magic[loc,11] .= 0
magic = convert(Array{Float64}, magic)
magic = [ones(n) magic]

data = DataFrame(magic)

Y = magic[:,12]
X = magic[:,1:11]
n, p = size(X)

function auc(y,ŷ)
    CP = sum(y)
    CN = length(y) - CP
    cutoff = ones(length(y), 100+1) * Diagonal(1:-1/100:0)
    pred = ŷ .> cutoff
    PP = vec(sum(pred, dims=1))
    TP = vec(sum((pred .+ y) .== 2, dims=1))
    FP = PP - TP
    TPR = TP ./ CP
    FPR = FP ./ CN
    H = 0.5 * (TPR[2:end] + TPR[1:end-1])
    W = diff(FPR)
    AUC = sum(H .* W)
    ACC = mean(pred[:,51] .== y)
    return AUC, ACC
end

N1 = 50
N2 = 50

record_AUC = readdlm("AUC.csv",',',Float64,'\n')
record_ACC = readdlm("ACC.csv",',',Float64,'\n')
#random, exact-A, apprx-A, exact-D, apprx_D, exact-A_K, apprx-A_K

for i = start:100
    v_auc = zeros(7)
    v_acc = zeros(7)
    #Random.seed!(1992)
    samp1 = sample(1:n, N1, replace = false)
    cand = setdiff(1:n, samp1)
    β = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp1,:], Binomial(), LogitLink()))
    p = logistic(X*β)
    w = p .* (1 .- p)

    ## Random Design
    samp2 = [samp1; sample(cand, N2, replace = false)]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[1], v_acc[1] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Exact Design A-opt
    ind = sagnol_A(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2; verbose=0, IC=1)
    samp2 = [samp1; cand[BitArray(round.(ind))]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[2], v_acc[2] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Approximate Design A-opt
    ind = sagnol_A(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2; verbose=0, IC=0)
    samp2 = [samp1; cand[rand(length(cand)) .< ind]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[3], v_acc[3] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Exact Design D-opt
    ind = sagnol_D(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2; verbose=0, IC=1)
    samp2 = [samp1; cand[BitArray(round.(ind))]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[4], v_acc[4] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Approximate Design D-opt
    ind = sagnol_D(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2; verbose=0, IC=0)
    samp2 = [samp1; cand[rand(length(cand)) .< ind]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[5], v_acc[5] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Exact Design A_K-opt
    ind = sagnol_A(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2;K = X', verbose=0, IC=1)
    samp2 = [samp1; cand[BitArray(round.(ind))]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[6], v_acc[6] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))

    ## Approximate Design A_K-opt
    ind = sagnol_A(Diagonal(sqrt.(w[cand]))*X[cand,:], Diagonal(sqrt.(w[samp1]))*X[samp1,:], N2;K=X', verbose=0, IC=0)
    samp2 = [samp1; cand[rand(length(cand)) .< ind]]
    β_new = coef(glm(@formula(x12 ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+0), data[samp2,:], Binomial(), LogitLink()))
    v_auc[7], v_acc[7] = auc(Y[setdiff(1:n, samp2)], logistic(X[setdiff(1:n, samp2),:]*β_new))
    
    record_AUC[i,:] = v_auc
    record_ACC[i,:] = v_acc
    
    println(i)
    writedlm("AUC.csv", record_AUC, ',')
    writedlm("ACC.csv", record_ACC, ',')
end