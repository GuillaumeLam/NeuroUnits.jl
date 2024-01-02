include("reservoir.jl")
# include("read_in.jl")
# include("read_out.jl")

struct LiquidStateMachine
    reservoir::Reservoir
    read_in_connections::Vector{Float64}  # Vector of neuron weight
    read_out_connections::Vector{Float64}  # Vector of neuron weight
    stim_timesteps::Int
    rest_timesteps::Int

    function LiquidStateMachine()
        reservoir = Reservoir()
        read_in_connections = [rand()>0.8 ? rand() : 0.0 for _ in 1:length(reservoir.neurons)]
        read_out_connections = [rand()>0.8 ? rand() : 0.0 for _ in 1:length(reservoir.neurons)]
        stim_timesteps = 100
        rest_timesteps = 100

        new(reservoir, read_in_connections, read_out_connections, stim_timesteps, rest_timesteps)
    end
end

function simulate_hist!(lsm::LiquidStateMachine, input::Vector{Float64})
    neuron_states, synapse_states, astrocyte_states = simulate_hist!(input, lsm.stim_timesteps, lsm.reservoir)
    neuron_statesp, synapse_statesp, astrocyte_statesp = simulate_hist!(zeros(length(lsm.reservoir.neurons)), lsm.rest_timesteps, lsm.reservoir)
    return hcat(neuron_states, neuron_statesp), hcat(synapse_states, synapse_statesp), hcat(astrocyte_states, astrocyte_statesp)
end

function simulate!(lsm::LiquidStateMachine, input::Vector{Float64})
    simulate!(input, lsm.stim_timesteps, lsm.reservoir)
    simulate!(zeros(length(lsm.reservoir.neurons)), lsm.rest_timesteps, lsm.reservoir)
end

function (lsm::LiquidStateMachine)(input::Vector{Float64})

    # h1 = lsm.read_in_connections(input)

    h2 = simulate!(lsm, h1)

    # output = lsm.read_out_connections(h2)

    return output
end

# function stimulate_neurons!(lsm::LiquidStateMachine, timestep::Int)
#     if timestep % (lsm.stim_timesteps + lsm.rest_timesteps) < lsm.stim_timesteps
#         for (neuron_idx, current) in lsm.read_in_connections
#             lsm.reservoir[neuron_idx].membrane_potential += current
#         end
#     end
# end

