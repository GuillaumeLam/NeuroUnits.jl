# Define abstract supertypes
abstract type AbstractNeuron end
abstract type AbstractSynapse end
abstract type AbstractAstrocyte end

mutable struct Neuron <: AbstractNeuron
    membrane_potential::Float64
    threshold::Float64
    reset_potential::Float64
    out_synapses::Vector{AbstractSynapse}
    in_synapses::Vector{AbstractSynapse}
    last_spike::Float64
    spike_train::Vector{Int}  # Binary spike train (1 for spike, 0 for no spike)
end

function initialize_neurons(num_neurons::Int, simulation_length::Int)
    neurons = Vector{Neuron}()
    for _ in 1:num_neurons
        neuron = Neuron(
            0.0,  # Rest voltage
            0.5,  # Example threshold
            0.0,  # Rest voltage after spike
            [],   # No synapses initially
            [],   # No synapses initially
            -Inf, # Initialize as if never spiked
            zeros(Int, simulation_length)  # Binary spike train
        )
        push!(neurons, neuron)
    end
    return neurons
end

function neuron_LIF_update!(neuron::Neuron; current_time::Float64, u_i::Float64, Δt::Float64=10.0)
    τ_v = 64.0
    θ_i = neuron.threshold
    spike_τ = 1.0

    # Update potential with input current
    neuron.membrane_potential += (-neuron.membrane_potential / τ_v + u_i) * Δt

    # Check for spikes and reset if necessary
    if neuron.membrane_potential >= θ_i
        neuron.membrane_potential = neuron.reset_potential
        neuron.spike_train[current_time] = 1  # Record spike at the current time index
        for s in neuron.out_synapses
            s.begin_syn_current += spike_τ
        end
    else
        neuron.spike_train[current_time] = 0  # No spike at the current time index
    end
end

neurons_LIF_update!(neurons::Vector{Neuron}; current_time::Float64, u_i::Vector{Float64}, Δt::Float64=10.0) =
    neuron_LIF_update!.(neurons; current_time, u_i, Δt)

mutable struct Synapse <: AbstractSynapse
    weight::Float64
    pre_neuron::AbstractNeuron
    post_neuron::AbstractNeuron
    T_pre::Float64  # Pre-synaptic trace
    T_post::Float64  # Post-synaptic trace
    begin_syn_current::Float64
    end_syn_current::Float64
    stdp_lr::Float64
end

# Function to initialize Synapses
function initialize_synapses(neurons::Vector{Neuron}, num_synapses::Int)
    synapses = Vector{Synapse}()
    for _ in 1:num_synapses

        pre_neuron = neurons[rand(1:length(neurons))]
        post_neuron = neurons[rand(1:length(neurons))]

        synapse = Synapse(
            # rand()<0.8 ? rand(0.7:0.1:1.0) : rand(-1.0:0.1:0-0.4),                # Random initial weight
            rand(),
            pre_neuron,
            post_neuron,
            -Inf,          # Initialize with no previous spikes
            -Inf,
            0.0,
            0.0,
            0.01               # Example STDP parameter
        )
        push!(synapses, synapse)
        push!(pre_neuron.out_synapses, synapse)
        push!(post_neuron.in_synapses, synapse)
    end
    return synapses
end

function synapse_STDP_update!(synapse::Synapse; current_time::Int, Δt::Float64=10.0)
    A_plus = 0.15
    A_minus = 0.15
    τ_plus = 10.0  # ms
    τ_minus = 10.0  # ms
    a_plus = 0.1
    a_minus = 0.1

    # Update traces based on the spike train
    T_pre_decay = exp(-Δt / τ_plus)
    T_post_decay = exp(-Δt / τ_minus)
    synapse.T_pre = synapse.T_pre * T_pre_decay + a_plus * synapse.pre_neuron.spike_train[current_time]
    synapse.T_post = synapse.T_post * T_post_decay + a_minus * synapse.post_neuron.spike_train[current_time]

    # STDP weight update based on the last spike times
    if synapse.pre_neuron.spike_train[current_time] == 1
        # Potentiation due to pre-synaptic spike
        synapse.weight += A_plus * synapse.T_pre * Δt
    end
    if synapse.post_neuron.spike_train[current_time] == 1
        # Depression due to post-synaptic spike
        synapse.weight -= A_minus * synapse.T_post * Δt
    end

    # Clamp weight within reasonable bounds
    synapse.weight = clamp(synapse.weight, -5.0, 5.0)
end

synapses_STDP_update!(synapses::Vector{Synapse}; current_time::Int, Δt::Float64=10.0) =
    synapse_STDP_update!.(synapses; current_time, Δt)

mutable struct Astrocyte <: AbstractAstrocyte
    A_astro::Float64
    τ_astro::Float64
    w_astro::Float64
    b_astro::Float64
    liquid_neurons::Vector{Neuron}
    input_neurons::Vector{Neuron}
end

function initialize_astrocytes(liquid_neurons::Vector{Neuron}, input_neurons::Vector{Neuron}, num_astrocytes::Int)
    astrocytes = Vector{Astrocyte}()
    for _ in 1:num_astrocytes
        astrocyte = Astrocyte(
            0.0,       # Initial A_astro value
            1.0,       # τ_astro should be set according to your model specifics
            1.0,       # w_astro, the weight for the astrocyte's influence
            0.15,      # b_astro, bias or base level of astrocyte's activity
            liquid_neurons,
            input_neurons
        )
        push!(astrocytes, astrocyte)
    end
    return astrocytes
end

function astrocyte_LIM_update!(astrocyte::Astrocyte; current_time::Int, Δt::Float64=10.0)
    # Calculate the total spikes from liquid and input neurons at the current time
    liquid_spikes = sum(neuron.spike_train[current_time] for neuron in astrocyte.liquid_neurons)
    input_spikes = sum(neuron.spike_train[current_time] for neuron in astrocyte.input_neurons)
    
    # Compute the change in astrocyte activity
    dA_astro_dt = (-astrocyte.A_astro / astrocyte.τ_astro) + astrocyte.w_astro * (liquid_spikes - input_spikes) + astrocyte.b_astro
    
    # Update the astrocyte's state using Euler integration
    astrocyte.A_astro += dA_astro_dt * Δt
end

astrocytes_LIM_update!(astrocytes::Vector{Astrocyte}; current_time::Int, Δt::Float64=10.0)=
    astrocyte_LIM_update!.(astrocytes; current_time, Δt)
