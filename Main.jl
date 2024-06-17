using  Flux, ProgressMeter, MLDatasets

include("Graph.jl")
include("RNN.jl")

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

seq_nmb = 4
seq_length = 196
neurons = 64
output = 10

RNN = define_RNN(seq_nmb, seq_length, neurons, output, 100)

batchsize = 100
α = 15e-3
epochs = 5
∇W = ∇(RNN)

test!(RNN, test_data)
for epoch in 1:epochs
    L = 0
    @time for (X_batch, Y_batch) in loader(train_data, batchsize=batchsize)

            @inbounds RNN.Xts[1].output = @views X_batch[1:196,:]
            @inbounds RNN.Xts[2].output = @views X_batch[197:392,:]
            @inbounds RNN.Xts[3].output = @views X_batch[393:588,:]
            @inbounds RNN.Xts[4].output = @views X_batch[589:end,:]

            @inbounds RNN.y.output = @views Y_batch[:,:]

            L += sum(forward!(RNN.L_RNN))
            
            backward!(RNN.L_RNN)

            acumulate_∇!(RNN, ∇W)
            reset!(RNN.L_RNN)
            
            update_batch∇!(RNN, ∇W, batchsize, α)
        end


    println("Current loss: ", L)
end

test!(RNN, test_data)