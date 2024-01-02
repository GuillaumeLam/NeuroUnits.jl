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
        read_in_connections = [rand()>0.9 for _ in 1:length(reservoir.neurons)]
        read_out_connections = [rand()>0.9 for _ in 1:length(reservoir.neurons)]
        stim_timesteps = 100
        rest_timesteps = 100

        new(reservoir, read_in_connections, read_out_connections, stim_timesteps, rest_timesteps)
    end

end

function simulate_hist!(lsm::LiquidStateMachine, duration::Int64)
    simulate_hist!(lsm.reservoir, duration)
end

function simulate!(lsm::LiquidStateMachine, duration::Float64)
    simulate!(lsm.reservoir, duration)
end

function (lsm::LiquidStateMachine)(input::Vector{Float64})
    # Simulate the reservoir
    simulate!(lsm, length(input))

    # Get the reservoir states
    reservoir_states = [neuron.state for neuron in lsm.reservoir.neurons]

    # Get the read out states
    read_out_states = [neuron.state for neuron in lsm.read_out_connections]

    # Return the read out states
    return read_out_states
end

# function stimulate_neurons!(lsm::LiquidStateMachine, timestep::Int)
#     if timestep % (lsm.stim_timesteps + lsm.rest_timesteps) < lsm.stim_timesteps
#         for (neuron_idx, current) in lsm.read_in_connections
#             lsm.reservoir[neuron_idx].membrane_potential += current
#         end
#     end
# end

