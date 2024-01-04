using Statistics

# Define abstract supertypes
abstract type AbstractNeuron end
abstract type AbstractSynapse end
abstract type AbstractAstrocyte end

# Neuron struct
mutable struct LIFNeuron <: AbstractNeuron
	membrane_potential::Float64
	threshold::Float64
	reset_potential::Float64
	spike_τ::Float64
	out_synapses::Vector{AbstractSynapse}  # Specify the concrete type for clarity
	in_synapses::Vector{AbstractSynapse}   # Specify the concrete type for clarity
	last_spike::Float64
	spike_train::Vector{Int}       # Binary spike train (1 for spike, 0 for no spike)
end

function initialize_neurons(num_neurons::Int; simulation_length::Int=100)
	neurons = Vector{LIFNeuron}()
	for _ in 1:num_neurons
		type = rand()<0.8 ? "excitatory" : "inhibitory"

		neuron = LIFNeuron(
			0.0,  # Rest voltage
			5.0,  # Example threshold
			-0.3,  # Rest voltage after spike
			type == "excitatory" ? 1.7 : -1.7,  # Spike amplitude
			[],   # No synapses initially
			[],   # No synapses initially
			-Inf, # Initialize as if never spiked
			zeros(Int, simulation_length)  # Binary spike train
		)
		push!(neurons, neuron)
	end
	return neurons
end

# LIF Neuron update function
function neuron_LIF_update!(neuron::N, current_time::Int, u_i::Float64, Δt::Float64) where {N <: AbstractNeuron}
	τ_v = 64.0
	θ_i = neuron.threshold
	absolute_refractory_period = 2.0  # Absolute refractory period duration in milliseconds

    # Check if the neuron is in the refractory period
    if !(current_time - neuron.last_spike < absolute_refractory_period)
		# Check for spikes and reset if necessary
		if neuron.membrane_potential >= θ_i
			neuron.membrane_potential = neuron.reset_potential
			neuron.spike_train[current_time] = 1  # Record spike at the current time index
			neuron.last_spike = current_time
		else
			# Update the membrane potential using Euler integration
			neuron.membrane_potential += (-neuron.membrane_potential + u_i) * (Δt / τ_v)
		end
	end
end

# LIF update function for all neurons
function neurons_LIF_update!(neurons::Vector{N}, current_time::Int, u_i::Vector{Float64}, Δt::Float64) where {N <: AbstractNeuron}
	u_i = u_i |> x -> [x; zeros(length(neurons) - length(x))]
	
	for (neuron, current) in zip(neurons, u_i)
		neuron_LIF_update!(neuron, current_time, current, Δt)
	end
end

# mutable struct SpikingNeuron <: AbstractNeuron
# 	spike_τ::Float64
# 	out_synapses::Vector{AbstractSynapse}  # Specify the concrete type for clarity
# 	last_spike::Float64
# 	spike_train::Vector{Int}       # Binary spike train (1 for spike, 0 for no spike)
# end

# Synapse struct
mutable struct Synapse <: AbstractSynapse
	weight::Float64
	weight_cap::Tuple{Float64, Float64}
	pre_neuron::AbstractNeuron           # Use Neuron type for direct access to properties
	post_neuron::AbstractNeuron          # Use Neuron type for direct access to properties
	T_pre::Float64               # Pre-synaptic trace
	T_post::Float64              # Post-synaptic trace
	delay_const::Float64
	delay_count::Float64
	spike_τ::Float64
	linked_astrocytes::Vector{AbstractAstrocyte}
end

# Function to initialize Synapses
function initialize_synapses(num_synapses::Int, neurons::Vector{N}) where {N <: AbstractNeuron}
	synapses = Vector{Synapse}()
	for _ in 1:num_synapses

		pre_neuron = rand(neurons)
		post_neuron = rand(neurons)

		# type = rand()<8 ? "excitatory" : "inhibitory"

		synapse = Synapse(
			rand(0:0.1:3),
			(0.0,3.0),
			pre_neuron,
			post_neuron,
			0.0,
			0.0,
			3.0,
			0.0,
			false,
			[]
		)
		push!(synapses, synapse)
		push!(pre_neuron.out_synapses, synapse)
		push!(post_neuron.in_synapses, synapse)
	end
	return synapses
end

# STDP Synapse update function
function synapse_STDP_update!(synapse::Synapse, current_time::Int, Δt::Float64)
	synapse.spike_τ = 0.0
	
	if synapse.linked_astrocytes != []
		A_minus = mean([astrocyte.A_astro for astrocyte in synapse.linked_astrocytes])
	else
		A_minus = 0.15
	end

	A_plus = 0.15
		
	τ_plus = 10.0  # ms
	τ_minus = 10.0  # ms
	a_plus = 0.1
	a_minus = 0.1

	# Transmit current to post-synaptic neuron
	if synapse.pre_neuron.spike_train[current_time] == 1
		synapse.delay_count = synapse.delay_const
	end

	if synapse.delay_count > 0
		synapse.delay_count -= 1.0
		if synapse.delay_count == 0 && !(current_time - synapse.post_neuron.last_spike < 2)
			synapse.post_neuron.membrane_potential += synapse.weight * synapse.pre_neuron.spike_τ
			synapse.spike_τ = synapse.weight * synapse.pre_neuron.spike_τ
		end
	end

	# # Update traces based on the spike train
	# T_pre_decay = exp(-Δt / τ_plus)
	# T_post_decay = exp(-Δt / τ_minus)

	# synapse.T_pre = synapse.T_pre * T_pre_decay + a_plus * synapse.pre_neuron.spike_train[current_time]
	# synapse.T_post = synapse.T_post * T_post_decay + a_minus * synapse.post_neuron.spike_train[current_time]

	# Update traces based on the spike train
	synapse.T_pre += (-synapse.T_pre + a_plus * synapse.pre_neuron.spike_train[current_time]) * (Δt / τ_plus)
	synapse.T_post += (-synapse.T_post + a_minus * synapse.post_neuron.spike_train[current_time]) * (Δt / τ_minus)

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
	synapse.weight = clamp(synapse.weight, -3.0, 3.0)
end

# STDP update function for all synapses
function synapses_STDP_update!(synapses::Vector{Synapse}, current_time::Int, Δt::Float64)
	for synapse in synapses
		synapse_STDP_update!(synapse, current_time, Δt)
	end
end


# Astrocyte struct
mutable struct Astrocyte <: AbstractAstrocyte
	A_astro::Float64
	τ_astro::Float64
	w_astro::Float64
	b_astro::Float64
	Γ_astro::Float64
	liquid_synapses::Vector{AbstractSynapse}
end
function initialize_astrocytes(num_astrocytes::Int, liquid_synapses::Vector{S}) where {S <: AbstractSynapse}
	astrocytes = Vector{Astrocyte}()
	for _ in 1:num_astrocytes
		modulated_synapses = rand(liquid_synapses, 100)
		# modulated_neurons = liquid_neurons

		astrocyte = Astrocyte(
			0.15,      	# Initial A_astro value
			1000.0,     	# τ_astro should be set according to your model specifics
			7.5e-3,       # w_astro, the weight for the astrocyte's influence
			0.01,      	# b_astro, bias or base level of astrocyte's activity
			0.9, 		# Γ_astro, the gain for the astrocyte's influence
			modulated_synapses
		)
		push!(astrocytes, astrocyte)
		for s in modulated_synapses
			push!(s.linked_astrocytes, astrocyte)
		end
	end
	return astrocytes
end

# Astrocyte LIM model update function
function astrocyte_LIM_update!(astrocyte::Astrocyte, current_time::Int, u_i::Vector{Float64}, Δt::Float64)
	# Calculate the total spikes from liquid and input neurons at the current time
	liquid_spikes = sum(synapse.spike_τ for synapse in astrocyte.liquid_synapses)
	input_spikes = sum(u_i)
	
	# Compute the change in astrocyte activity
	dA_astro_dt = (-astrocyte.A_astro * astrocyte.Γ_astro + astrocyte.w_astro * (liquid_spikes - input_spikes) + astrocyte.b_astro) / astrocyte.τ_astro
	
	# Update the astrocyte's state using Euler integration
	astrocyte.A_astro += dA_astro_dt * Δt
end

# LIM update function for all astrocytes
function astrocytes_LIM_update!(astrocytes::Vector{Astrocyte}, current_time::Int, u_i::Vector{Float64}, Δt::Float64)
	s = []
	println("Sum of A_astro before: ", mean([astrocyte.A_astro for astrocyte in astrocytes]))
	for astrocyte in astrocytes
		liquid_spikes = sum(synapse.spike_τ for synapse in astrocyte.liquid_synapses)
		input_spikes = sum(u_i)
		push!(s, liquid_spikes - input_spikes)

		astrocyte_LIM_update!(astrocyte, current_time, u_i, Δt)
	end
	println("Sum of A_astro after: ", mean([astrocyte.A_astro for astrocyte in astrocytes]))

	println("Astrocyte activity; mean liq-in spike diff: ", mean(s))
end

function Base.show(io::IO, ::MIME"text/plain", n::Vector{AbstractNeuron})
    println(io, "Neurons.")
end

function Base.show(io::IO, ::MIME"text/plain", s::Vector{AbstractSynapse})
	println(io, "Synapses!")
end

function Base.show(io::IO, ::MIME"text/plain", a::Vector{AbstractAstrocyte})
	println(io, "Astrocytes!!")
end