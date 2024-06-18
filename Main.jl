using  Flux, ProgressMeter, MLDatasets

include("Graph.jl")
include("RNN.jl")

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

α = 15e-3
epochs = 5
output = 10
seq_nmb = 4
neurons = 64
batchsize = 100
seq_length = 196

RNN = define_RNN(seq_nmb, seq_length, neurons, output, batchsize)
test🔍!(RNN, test_data, train_data)

for epoch in 1:epochs
    @inbounds @time for (X_batch, Y_batch) in loader(train_data, batchsize=batchsize)
        RNN.Xts[1].output = @views X_batch[1:196,:]
        RNN.Xts[2].output = @views X_batch[197:392,:]
        RNN.Xts[3].output = @views X_batch[393:588,:]
        RNN.Xts[4].output = @views X_batch[589:end,:]

        RNN.y.output = @views Y_batch

        forward!(RNN.L_RNN)
        backward!(RNN.L_RNN)
        
        update_batch∇!(RNN, batchsize, α)
        reset!(RNN.L_RNN)
    end

    test🔍!(RNN, test_data, train_data)
end