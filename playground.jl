using Pkg

Pkg.activate("./LSM.jl")

include("lsm.jl")
include("graphics.jl")

# Initialize components
@time lsm = LiquidStateMachine()
hist = Dict(
    "n" => Matrix{Float64}(undef,size(lsm.neurons,1),0), 
    "s" => Matrix{Float64}(undef,size(lsm.synapses,1),0), 
    "a" => Matrix{Float64}(undef,size(lsm.astrocytes,1),0)
)
coin() = rand()>0.5 ? 1.0 : 0.0

@time for _ in 1:30
    # input_current = [Float64(rand()>0.7) for _ in 1:10]
    input_current = [ 0.0, coin(), 0, 0, 0, 0, 0, 0, 0, 0]

    # Simulate the reservoir for the given number of timesteps
    neuron_states, synapse_states, astrocyte_states = simulate_hist!(lsm, input_current, stim_timesteps=25, rest_timesteps=0)
    hist["n"] = hcat(hist["n"], neuron_states)
    hist["s"] = hcat(hist["s"], synapse_states)
    hist["a"] = hcat(hist["a"], astrocyte_states)
end

# Call the function with the states and duration
@time create_plots(hist["n"], hist["s"], hist["a"])
