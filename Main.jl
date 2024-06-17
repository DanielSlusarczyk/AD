using  Flux, ProgressMeter, MLDatasets

include("Graph.jl")
include("RNN.jl")

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

seq_nmb = 4
seq_length = 196
neurons = 64
output = 10

RNN = define_RNN(seq_nmb, seq_length, neurons, output)

batchsize = 100
α = 15e-3
epochs = 5
∇W = ∇(RNN)

test!(RNN, test_data)
for epoch in 1:epochs
    L = 0
    @time for (X_batch, Y_batch) in loader(train_data, batchsize=batchsize)
        for i in 1:batchsize
            @inbounds RNN.Xts[1].output = X_batch[1:196, i:i]
            @inbounds RNN.Xts[2].output = X_batch[197:392, i:i]
            @inbounds RNN.Xts[3].output = X_batch[393:588, i:i]
            @inbounds RNN.Xts[4].output = X_batch[589:end, i:i]

            @inbounds RNN.y.output = Y_batch[:,i:i]

            L += forward!(RNN.L_RNN)[1]
            backward!(RNN.L_RNN)

            acumulate_∇!(RNN, ∇W)
            reset!(RNN.L_RNN)
        end

        update_batch∇!(RNN, ∇W, batchsize, α)
    end

    println("Current loss: ", L)
end

test!(RNN, test_data)