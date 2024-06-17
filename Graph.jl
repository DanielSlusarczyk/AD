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
    inputs :: Matrix{Float64}
    output :: Matrix{Float64}
    gradient :: Any
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
    for node in order
        reset!(node)
    end
end

init!(node::Constant) = nothing
init!(node::GraphNode) = node.gradient = zeros(size(node.output))

function init!(order::Vector)
    for node in order
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
    for node in order
        forward!(node)
        
    end
    return last(order).output
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = fill(seed, 1, 1)
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

backward!(node::Constant) = nothing
backward!(node::Variable) = nothing
backward!(node::Operator) = backward!(node, [input.output for input in node.inputs]..., node.gradient)

import Base: -
-(x::GraphNode) = ScalarOperator(-, x)
forward!(::ScalarOperator{typeof(-)}, x) = let 
    node.output .= -x
end
forward_init!(::ScalarOperator{typeof(-)}, x) = let 
    node.output = -x
end
backward!(::ScalarOperator{typeof(-)}, ∇) = let     
    node.inputs[1].gradient .+= -∇
end

σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward!(node::BroadcastedOperator{typeof(σ)}, x) = let 
    node.output .= exp.(x) ./ sum(exp.(x))
end
forward_init!(node::BroadcastedOperator{typeof(σ)}, x) = let 
    node.output = exp.(x) ./ sum(exp.(x))
end

CSLoss(y::GraphNode, ŷ::GraphNode) = BroadcastedOperator(CSLoss, y, ŷ)
forward!(node::BroadcastedOperator{typeof(CSLoss)}, y, ŷ) = let    
    σ = exp.(ŷ) ./ sum(exp.(ŷ))

    node.output = fill(-sum(y .* log.(σ)), 1, 1)
end
forward_init!(node::BroadcastedOperator{typeof(CSLoss)}, y, ŷ) = let    
    σ = exp.(ŷ) ./ sum(exp.(ŷ))

    node.output = fill(-sum(y .* log.(σ)), 1, 1)
end
backward!(node::BroadcastedOperator{typeof(CSLoss)}, y, ŷ, ∇) = let

    # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    σ = exp.(ŷ) ./ sum(exp.(ŷ))

    node.inputs[1].gradient += y
    node.inputs[2].gradient .+= (σ .- y) .* ∇
end

import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward!(node::BroadcastedOperator{typeof(mul!)}, A, x) = let 
    mul!(node.output, A, x)
end
forward_init!(node::BroadcastedOperator{typeof(mul!)}, A, x) = let     
    node.output = A * x
end
backward!(node::BroadcastedOperator{typeof(mul!)}, A, x, ∇) = let 
    mul!(node.inputs[1].gradient, ∇, x', 1, 1)

    mul!(node.inputs[2].gradient, A', ∇, 1, 1)
end

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward!(node::BroadcastedOperator{typeof(*)}, A, x) = let     
    node.output .= A .* x
end
forward_init!(node::BroadcastedOperator{typeof(*)}, A, x) = let     
    node.output = A .* x
end
backward!(node::BroadcastedOperator{typeof(*)}, A, B, ∇) = let
    node.inputs[1].gradient .+= B .* ∇

    node.inputs[2].gradient .+= A .* ∇
end
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward!(node::BroadcastedOperator{typeof(+)}, A, B) = let 
    node.output .= A .+ B
end
forward_init!(node::BroadcastedOperator{typeof(+)}, A, B) = let     
    node.output = A .+ B
end
backward!(node::BroadcastedOperator{typeof(+)}, _, _, ∇) = let 
    node.inputs[1].gradient .+= ∇
    node.inputs[2].gradient .+= ∇
end
Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, x)
forward!(node::BroadcastedOperator{typeof(tanh)}, x) = let     
    node.output .= tanh.(x)
end
forward_init!(node::BroadcastedOperator{typeof(tanh)}, x) = let     
    node.output = tanh.(x)
end
backward!(node::BroadcastedOperator{typeof(tanh)}, x, ∇) = let
    node.inputs[1].gradient .+=  (1 .- tanh.(x) .^ 2) .* ∇
end