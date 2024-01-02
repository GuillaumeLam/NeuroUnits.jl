# Define abstract supertypes
abstract type AbstractNeuron end
abstract type AbstractSynapse end
abstract type AbstractAstrocyte end

mutable struct Neuron <: AbstractNeuron
    membrane_potential::Float64
    threshold::Float64
    reset_potential::Float64
    decay_factor::Float64
    out_synapses::Vector{AbstractSynapse}
    last_spike::Float64
end

# Function to initialize Neurons
function initialize_neurons(num_neurons::Int)
    neurons = Vector{Neuron}()
    for _ in 1:num_neurons
        neuron = Neuron(
            0.0,  # Rest voltage
            0.85,          # Example threshold
            0.0,    # Rest voltage after spike
            0.95,       # Example decay factor
            [],          # No synapses initially
            -Inf
        )
        push!(neurons, neuron)
    end
    return neurons
end

function update_neuron(current_time, neuron::Neuron, input_current::Float64=0.0; spike_current::Float64=1.0)
    neuron.membrane_potential += input_current

    if neuron.membrane_potential > neuron.threshold
        neuron.membrane_potential = neuron.reset_potential
        for synapse in neuron.out_synapses
            synapse.post_neuron.membrane_potential += synapse.weight * spike_τ
        end
        neuron.last_spike = current_time
        return true
    else
        neuron.membrane_potential *= neuron.decay_factor
        return false
    end
end

function update_neurons(t, neurons::Vector{Neuron}, input_currents::Vector{Float64}; spike_τ::Float64=1.0)
    spikes = Vector{Neuron}()

    for (n, i) in zip(neurons, input_currents)
        n.membrane_potential += i
        if n.membrane_potential > n.threshold
            n.membrane_potential = n.reset_potential
            for s in n.out_synapses
                s.post_neuron.membrane_potential += s.weight * spike_τ
            end
            n.last_spike = t
            push!(spikes, n)
        else
            n.membrane_potential *= n.decay_factor
        end
    end
    
    return spikes
end

mutable struct Synapse <: AbstractSynapse
    weight::Float64
    pre_neuron::AbstractNeuron
    post_neuron::AbstractNeuron
    last_pre_spike::Float64
    last_post_spike::Float64
    stdp_param::Float64
end

# Function to initialize Synapses
function initialize_synapses(neurons::Vector{Neuron}, num_synapses::Int)
    synapses = Vector{Synapse}()
    for _ in 1:num_synapses

        pre_neuron = neurons[rand(1:length(neurons))]
        post_neuron = neurons[rand(1:length(neurons))]

        synapse = Synapse(
            rand(),                # Random initial weight
            pre_neuron,
            post_neuron,
            -Inf,          # Initialize with no previous spikes
            -Inf,
            0.01               # Example STDP parameter
        )
        push!(synapses, synapse)
        push!(pre_neuron.out_synapses, synapse)
    end
    return synapses
end

function update_synapses(t, synapses::Vector{Synapse}, spikes::Vector{Neuron})
    for synapse in synapses
        for neuron in spikes
            if neuron == synapse.pre_neuron
                synapse.last_pre_spike = t
            end
            if neuron == synapse.post_neuron
                synapse.last_post_spike = t
            end
        end

        dt = synapse.last_post_spike - synapse.last_pre_spike
        if dt > 0
            synapse.weight += synapse.stdp_param * exp(-abs(dt))
        elseif dt < 0
            synapse.weight -= synapse.stdp_param * exp(-abs(dt))
        end

        # Ensuring the weight stays within reasonable bounds
        synapse.weight = clamp(synapse.weight, 0.0, 1.0)
    end
end

mutable struct Astrocyte <: AbstractAstrocyte
    modulation_factor::Float64
    threshold::Float64
    neurons::Array{AbstractNeuron}
    synapses::Array{AbstractSynapse}
end

# Function to initialize Astrocytes
function initialize_astrocytes(neurons::Vector{Neuron}, num_astrocytes::Int)
    astrocytes = Vector{Astrocyte}()
    for _ in 1:num_astrocytes
        astrocyte = Astrocyte(
            0.05,      # Example modulation factor
            5.0,               # Example threshold for activation
            neurons,             # All neurons for simplicity
            []                  # No synapses for this example
        )
        push!(astrocytes, astrocyte)
    end
    return astrocytes
end

function update_astrocyte(astrocyte::Astrocyte)
    # Calculate the overall activity in the vicinity of the astrocyte
    local_activity = sum([neuron.membrane_potential for neuron in astrocyte.neurons])

    # If the activity exceeds a certain threshold, modulate the properties of neurons and synapses
    if local_activity > astrocyte.threshold
        for neuron in astrocyte.neurons
            neuron.threshold += astrocyte.modulation_factor
        end
        for synapse in astrocyte.synapses
            synapse.weight *= (1 + astrocyte.modulation_factor)
        end
    end
end

function simulate_hist!(input_currents::Vector{Float64}, duration::Int64, reservoir_neurons::Vector{Neuron}, reservoir_synapses::Vector{Synapse}, reservoir_astrocytes::Vector{Astrocyte})
    # Initialize states for neurons and synapses
    neuron_states = zeros(length(neurons), duration)
    synapse_states = zeros(length(synapses), duration)
    astrocyte_states = zeros(length(astrocytes), duration)

    for t in 1:duration
        spikes = update_neurons(t, reservoir_neurons, input_currents)
        neuron_states[:, t] = [neuron.membrane_potential for neuron in reservoir_neurons]

        update_synapses(t, reservoir_synapses, spikes)
        synapse_states[:, t] = [synapse.weight for synapse in reservoir_synapses]

        # Update astrocytes based on the overall activity
        for (a, astrocyte) in enumerate(reservoir_astrocytes)
            update_astrocyte(astrocyte)
            astrocyte_states[a, t] = astrocyte.modulation_factor
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

struct Reservoir
    neurons::Vector{Neuron}
    synapses::Array{Synapse}
    astrocytes::Array{Astrocyte}

    function Reservoir(;num_neurons::Int=10, num_synapses::Int=20, num_astrocytes::Int=2)
        neurons = initialize_neurons(num_neurons)
        synapses = initialize_synapses(neurons, num_synapses)
        astrocytes = initialize_astrocytes(neurons, num_astrocytes)
        new(neurons, synapses, astrocytes)
    end

    function Reservoir(neurons::Vector{Neuron}, synapses::Array{Synapse}, astrocytes::Array{Astrocyte})
        new(neurons, synapses, astrocytes)
    end
end

function simulate_hist!(input_currents::Vector{Float64}, duration::Int64, reservoir::Reservoir)
    simulate_hist!(input_currents, duration, reservoir.neurons, reservoir.synapses, reservoir.astrocytes)
end

function simulate!(input_currents::Vector{Float64}, duration::Int64, reservoir::Reservoir)
    simulate!(input_currents, duration, reservoir.neurons, reservoir.synapses, reservoir.astrocytes)
end

