mutable struct RNNet
    Wxh
    Whh
    Why
    bh
    by

    Xts
    h
    y
    
    L_RNN
    ŷ_RNN
    
    batchsize

    RNNet() = new(
        ntuple(x->nothing, fieldcount(RNNet))...
    )
end

mutable struct ∇
    ∇Whh
    ∇Wxh
    ∇Why
    ∇bh
    ∇by

    ∇(net::RNNet) = new(
        zeros(size(net.Whh.output)),
        zeros(size(net.Wxh.output)),
        zeros(size(net.Why.output)),
        zeros(size(net.bh.output)),
        zeros(size(net.by.output))
    )
end

function loader(data; batchsize::Int=1)
    x1dim = convert(Matrix{Float64}, reshape(data.features, 28 * 28, :)) # reshape 28×28 pixels into a vector of pixels
    yhot  = Flux.onehotbatch(data.targets, 0:9) # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

function init!(net, n_input::Int64, n_neurons::Int64, n_output::Int64, batchsize::Int64)

    # Initialization using Xavier method
    # https://fluxml.ai/Flux.jl/stable/utilities/

    net.Wxh = Variable(Flux.glorot_normal(n_neurons, n_input), name="Wxh", reset=false)
    net.Whh = Variable(Flux.glorot_normal(n_neurons, n_neurons), name="Whh", reset=false)
    net.Why = Variable(Flux.glorot_normal(n_output, n_neurons), name="Why", reset=false)
    net.bh = Variable(Flux.glorot_normal(n_neurons, batchsize), name="bh", reset=false)
    net.by = Variable(Flux.glorot_normal(n_output, batchsize), name="by", reset=false)
    net.h = Variable(Flux.glorot_normal(n_neurons, batchsize), name="h_prev", reset=false)
    
end

# Create hidden layer
function RNNDense(Xt, Wxh, Whh, bh, h_prev)
    h = tanh.(Wxh * Xt .+ Whh * h_prev .+ bh)
    h.name = "h"
    return h
end

# Create output layer
function RNNDense(h, Wyh, by)
    ŷ = Wyh * h .+ by
    ŷ.name = "ŷ"
    return ŷ
end

# Cross entropy loss layer
function RRNDense(ŷ, y)
    L = CSLoss(y, ŷ)
    L.name = "L"
    return L
end

# Create recurrent connections for each sequence
function model!(net::RNNet, seqs::Int64, n_input::Int64, batchsize::Int64)

    Xts = Vector()
    for _ in 1:seqs
        Xt = Variable(randn(n_input, batchsize), name="Xt")
        h = RNNDense(Xt, net.Wxh, net.Whh, net.bh, net.h)
        
        push!(Xts, Xt)
        net.h = h
    end

    net.Xts = Xts
end

function model_output!(net::RNNet, output::Int64, batchsize::Int64)
    net.y = Variable(randn(output, batchsize), name="y")

    ŷ = RNNDense(net.h, net.Why, net.by)
    L = RRNDense(ŷ, net.y)

    net.L_RNN = topological_sort(L)
    net.ŷ_RNN = topological_sort(σ(ŷ))
end

function test!(model, data)
    correct = 0
    
    for (X_batch, Y_batch) in loader(data, batchsize=model.batchsize)


            model.Xts[1].output = @views X_batch[1:196, :]
            model.Xts[2].output = @views X_batch[197:392, :]
            model.Xts[3].output = @views X_batch[393:588, :]
            model.Xts[4].output = @views X_batch[589:end, :]
    
            y = @views Y_batch[:,:]
            ŷ = forward!(model.ŷ_RNN)

            for i in axes(y, 2)
                if Flux.onecold(ŷ[:, i]) == Flux.onecold(y[:, i])
                    correct += 1
                end
            end
    end

    println("Correct: ", round(100 * correct/length(data); digits=2), "%")
end

function acumulate_∇!(net::RNNet, ∇W::∇)
    @. ∇W.∇Whh += net.Whh.gradient
    @. ∇W.∇Wxh += net.Wxh.gradient
    @. ∇W.∇Why += net.Why.gradient
    @. ∇W.∇bh += net.bh.gradient
    @. ∇W.∇by += net.by.gradient
end

function update_batch∇!(net::RNNet, ∇W::∇, batch::Int64, α = 0.01)
    @. net.Whh.output -= α * ∇W.∇Whh / batch
    fill!(∇W.∇Whh, 0.)
    
    @. net.Wxh.output -= α * ∇W.∇Wxh / batch
    fill!(∇W.∇Wxh, 0.)

    @. net.Why.output -= α * ∇W.∇Why / batch
    fill!(∇W.∇Why, 0.)

    @. net.bh.output -= α * ∇W.∇bh / batch
    fill!(∇W.∇bh, 0.)

    @. net.by.output -= α * ∇W.∇by / batch
    fill!(∇W.∇by, 0.)
end

function define_RNN(sequence::Int64, length::Int64, neurons::Int64, output::Int64, batchsize::Int64)
    net = RNNet()
    
    init!(net, length, neurons, output, batchsize)
    
    model!(net, sequence, length, batchsize)
    
    model_output!(net, output, batchsize)



    init!(net.L_RNN)
    init!(net.ŷ_RNN)

    net.batchsize = batchsize

    return net
end

