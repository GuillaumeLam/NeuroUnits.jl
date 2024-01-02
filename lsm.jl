include("reservoir.jl")
# include("read_in.jl")
# include("read_out.jl")

struct LiquidStateMachine
    neurons::Vector{Neuron}
    synapses::Array{Synapse}
    astrocytes::Array{Astrocyte}
    read_in_connections::Vector{Float64}  # Vector of neuron weight
    read_out_connections::Vector{Float64}  # Vector of neuron weight

    function LiquidStateMachine(;num_neurons::Int=10, num_synapses::Int=30, num_astrocytes::Int=10)
        neurons = initialize_neurons(num_neurons)
        synapses = initialize_synapses(neurons, num_synapses)
        astrocytes = initialize_astrocytes(neurons, synapses, num_astrocytes)
        read_in_connections = [rand()>0.8 ? rand() : 0.0 for _ in 1:length(neurons)]
        read_out_connections = [rand()>0.8 ? rand() : 0.0 for _ in 1:length(neurons)]

        new(neurons, synapses, astrocytes, read_in_connections, read_out_connections)
    end
end

function simulate_hist!(lsm::LiquidStateMachine, input::Vector{Float64}; stim_timesteps::Int=100, rest_timesteps::Int=50)
    neuron_states, synapse_states, astrocyte_states = simulate_hist!(input, stim_timesteps, lsm.neurons, lsm.synapses, lsm.astrocytes)
    neuron_statesp, synapse_statesp, astrocyte_statesp = simulate_hist!(zeros(length(lsm.neurons)), rest_timesteps, lsm.neurons, lsm.synapses, lsm.astrocytes)
    return hcat(neuron_states, neuron_statesp), hcat(synapse_states, synapse_statesp), hcat(astrocyte_states, astrocyte_statesp)
end

function simulate!(lsm::LiquidStateMachine, input::Vector{Float64}; stim_timesteps::Int=100, rest_timesteps::Int=50)
    simulate!(input, stim_timesteps, lsm.neurons, lsm.synapses, lsm.astrocytes)
    simulate!(zeros(length(lsm.neurons)), rest_timesteps, lsm.neurons, lsm.synapses, lsm.astrocytes)
end

function simulate_hist!(input_currents::Vector{Float64}, duration::Int64, neurons::Vector{Neuron}, synapses::Vector{Synapse}, astrocytes::Vector{Astrocyte})
    # Initialize states for neurons and synapses
    neuron_states = zeros(length(neurons), duration)
    synapse_states = zeros(length(synapses), duration)
    astrocyte_states = zeros(length(astrocytes), duration)

    for t in 1:duration
        spikes = update_neurons(t, neurons, input_currents)
        neuron_states[:, t] = [neuron.membrane_potential for neuron in neurons]

        update_synapses(t, synapses, spikes)
        synapse_states[:, t] = [synapse.weight for synapse in synapses]

        for (a, astrocyte) in enumerate(astrocytes)
            update_astrocyte(astrocyte)
            astrocyte_states[a, t] = astrocyte.local_activity
        end
    end

    return neuron_states, synapse_states, astrocyte_states
end

function simulate!(input_currents::Vector{Float64}, duration::Float64, reservoir_neurons::Vector{Neuron}, reservoir_synapses::Array{Synapse}, reservoir_astrocytes::Array{Astrocyte})
    for t in 1:duration
        for (n, neuron) in enumerate(reservoir_neurons)
            update_neuron(neuron, input_currents[n])
        end

        for synapse in reservoir_synapses
            update_synapse(synapse, t)
        end

        for astrocyte in reservoir_astrocytes
            update_astrocyte(astrocyte)
        end
    end
end

function (lsm::LiquidStateMachine)(input::Vector{Float64})

    # h1 = lsm.read_in_connections(input)

    h2 = simulate!(lsm, input)

    # output = lsm.read_out_connections(h2)

    return h2
end

