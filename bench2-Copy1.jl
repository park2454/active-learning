include("functions.jl")
using GLM, DataFrames, Statistics, DelimitedFiles, Random, Dates, LinearAlgebra, BenchmarkTools, JLD2, FileIO
import StatsBase.sample

Ncand = [10,20,50,100,200,500,1000,2000,5000,10000]
Np = [10,20,50,100,200,500,1000,2000,5000,10000]

mosek_elapsed = zeros(Float64, 10,10)
mosek_optval = zeros(Float64, 10,10)

Random.seed!(1992)

for j = [1,4]
    for i = 5
        X = randn(Ncand[i], Np[j])
        reg = sqrt(0.1)*Diagonal(ones(Np[j]))
        k = Int(0.2*Ncand[i])
        w = sagnol_A(X, reg, k; K=X', verbose=0, IC=0)
        mosek_optval[i,j] = tr(X*((X'*Diagonal(w)*X)\(X')))
        mosek_elapsed[i,j] = @elapsed sagnol_A(X, reg, k; K=X', verbose=0, IC=0)
        println(mosek_elapsed[i,j])
    end
end
