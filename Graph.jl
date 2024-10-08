using LinearAlgebra
abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Matrix{Float64}
    gradient :: Matrix{Float64}
    name :: String
    reset :: Bool
    Variable(output; name="?", reset=true) = new(output, [;;], name, reset)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Matrix{Float64}
    gradient :: Matrix{Float64}
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, [;;], [;;], name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Matrix{Float64}
    gradient :: Matrix{Float64}
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, [;;], [;;], name)
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::GraphNode) = fill!(node.gradient, 0.)
function reset!(order::Vector)
    @inbounds for node in order
        reset!(node)
    end
end

init!(node::Constant) = nothing
init!(node::GraphNode) = node.gradient = zeros(size(node.output))

function init!(order::Vector)
    @inbounds for node in order
        forward_init!(node)
        init!(node)
    end
end

forward!(node::Constant) = nothing
forward!(node::Variable) = nothing
forward!(node::Operator) = forward!(node, [input.output for input in node.inputs]...)

forward_init!(node::Constant) = nothing
forward_init!(node::Variable) = nothing
forward_init!(node::Operator) = forward_init!(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    @inbounds for node in order
        forward!(node)
        
    end
    return last(order).output
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = fill(seed, 1, length(result.output))
    @inbounds for node in reverse(order)
        backward!(node)
    end
    return nothing
end

backward!(node::Constant) = nothing
backward!(node::Variable) = nothing
backward!(node::Operator) = backward!(node, [input.output for input in node.inputs]..., node.gradient)

σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward!(node::BroadcastedOperator{typeof(σ)}, x::Matrix{Float64}) = let 
    node.output .= exp.(x) ./ sum(exp.(x), dims=1)
end
forward_init!(node::BroadcastedOperator{typeof(σ)}, x::Matrix{Float64}) = let 
    node.output = exp.(x) ./ sum(exp.(x), dims=1)
end

CSLoss(y::GraphNode, ŷ::GraphNode) = BroadcastedOperator(CSLoss, y, ŷ)
forward!(node::BroadcastedOperator{typeof(CSLoss)}, y::Matrix{Float64}, ŷ::Matrix{Float64}) = let    
    σ = exp.(ŷ) ./ (sum(exp.(ŷ), dims=1))

    node.output = -sum(y .* log.(σ), dims=1)
end
forward_init!(node::BroadcastedOperator{typeof(CSLoss)}, y::Matrix{Float64}, ŷ::Matrix{Float64}) = let    
    σ = exp.(ŷ) ./ sum(exp.(ŷ))
    node.output = -sum(y .* log.(σ), dims=1)
end
backward!(node::BroadcastedOperator{typeof(CSLoss)}, y::Matrix{Float64}, ŷ::Matrix{Float64}, ∇::Matrix{Float64}) = let

    # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    σ = exp.(ŷ) ./ sum(exp.(ŷ), dims=1)

    node.inputs[1].gradient += y
    node.inputs[2].gradient .+= (σ .- y) .* ∇
end
import Base: +
+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
forward!(node::ScalarOperator{typeof(+)}, A::Matrix{Float64}, B::Matrix{Float64}) = let 
    node.output .= A .+ B
end
forward_init!(node::ScalarOperator{typeof(+)}, A::Matrix{Float64}, B::Matrix{Float64}) = let     
    node.output = A .+ B
end
backward!(node::ScalarOperator{typeof(+)}, _::Matrix{Float64}, _::Matrix{Float64}, ∇::Matrix{Float64}) = let 
    node.inputs[1].gradient .+= ∇
    node.inputs[2].gradient .+= ∇
end

import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, B::GraphNode) = BroadcastedOperator(mul!, A, B)
forward!(node::BroadcastedOperator{typeof(mul!)}, A::Matrix{Float64}, B::Matrix{Float64}) = let 
    mul!(node.output, A, B)
end
forward_init!(node::BroadcastedOperator{typeof(mul!)}, A::Matrix{Float64}, B::Matrix{Float64}) = let     
    node.output = A * B
end
backward!(node::BroadcastedOperator{typeof(mul!)}, A::Matrix{Float64}, B::Matrix{Float64}, ∇::Matrix{Float64}) = let 
    mul!(node.inputs[1].gradient, ∇, B', 1, 1)

    mul!(node.inputs[2].gradient, A', ∇, 1, 1)
end

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, A::GraphNode, B::GraphNode) = BroadcastedOperator(*, A::Matrix{Float64}, B::Matrix{Float64})
forward!(node::BroadcastedOperator{typeof(*)}, A::Matrix{Float64}, B::Matrix{Float64}) = let     
    node.output .= A .* B
end
forward_init!(node::BroadcastedOperator{typeof(*)}, A::Matrix{Float64}, B::Matrix{Float64}) = let     
    node.output = A .* B
end
backward!(node::BroadcastedOperator{typeof(*)}, A::Matrix{Float64}, B::Matrix{Float64}, ∇::Matrix{Float64}) = let
    node.inputs[1].gradient .+= B .* ∇

    node.inputs[2].gradient .+= A .* ∇
end
Base.Broadcast.broadcasted(+, A::GraphNode, B::GraphNode) = BroadcastedOperator(+, A, B)
forward!(node::BroadcastedOperator{typeof(+)}, A::Matrix{Float64}, B::Matrix{Float64}) = let 
    node.output .= A .+ B
end
forward_init!(node::BroadcastedOperator{typeof(+)}, A::Matrix{Float64}, B::Matrix{Float64}) = let     
    node.output = A .+ B
end
backward!(node::BroadcastedOperator{typeof(+)}, _::Matrix{Float64}, _::Matrix{Float64}, ∇::Matrix{Float64}) = let 
    node.inputs[1].gradient .+= ∇
    node.inputs[2].gradient .+= sum(∇, dims=2)
end
Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, x)
forward!(node::BroadcastedOperator{typeof(tanh)}, x::Matrix{Float64}) = let     
    node.output .= tanh.(x)
end
forward_init!(node::BroadcastedOperator{typeof(tanh)}, x::Matrix{Float64}) = let     
    node.output = tanh.(x)
end
backward!(node::BroadcastedOperator{typeof(tanh)}, x::Matrix{Float64}, ∇::Matrix{Float64}) = let
    node.inputs[1].gradient .+=  (1 .- tanh.(x) .^ 2) .* ∇
end