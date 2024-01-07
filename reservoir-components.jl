using Distributions
using DSP
using Random
using Statistics
include("util.jl")

abstract type AbstractNeuron end
abstract type AbstractSynapse end
abstract type AbstractAstrocyte end

mutable struct LiquidNeuron <: AbstractNeuron
	membrane_potential::Float64
	threshold::Float64
	spike_τ::Float64
	out_synapses::Vector{AbstractSynapse}
	in_synapses::Vector{AbstractSynapse}
	last_spike::Float64
	spike_train::Vector{Int}
	position::Tuple{Float64, Float64, Float64}
	linked_astrocytes::Vector{AbstractAstrocyte}
end

function initialize_neurons_on_grid(grid_positions::Vector{Tuple{Float64, Float64, Float64}}, num_neurons::Int; simulation_length::Int=100)
	if length(grid_positions) < num_neurons
		error("Not enough unique grid positions for the number of neurons.")
	end
	shuffled_positions = shuffle(grid_positions)
	neurons = Vector{LiquidNeuron}()
	for (_, position) in zip(1:num_neurons, shuffled_positions)
		type = rand()<0.8 ? "excitatory" : "inhibitory"
		neuron = LiquidNeuron(0.0, 20.0, type == "excitatory" ? 1.0 : -1.0, [], [], -Inf, [], position, [])
		push!(neurons, neuron)
	end
	return neurons
end

δ(t) = t == 0 ? 1 : 0
H(t) = t >= 0 ? 1 : 0
α_u_t(t, τ_u=1.0) = exp(-t / τ_u) * H(t)

function neuron_LIF_update!(neuron::N, current_time::Int, spike_wagon::Float64, Δt::Float64; absolute_refractory_period::Int=2) where {N <: AbstractNeuron}
	
	# if current_time - neuron.last_spike < absolute_refractory_period
	# 	σ_i = 0.0
	# else
	# 	σ_i = v_i

	# 	for syn in neuron.in_synapses
	# 		σ_i += syn.spike_τ
	# 	end
	# end

	σ_i = 0.0

	for syn in neuron.in_synapses
		σ_i += syn.spike_τ
	end
	
	τ_v = 64.0
	θ_i = neuron.threshold
	b_i = 0.0
	
	u_i = σ_i + b_i

	internal_spike = neuron.membrane_potential>=θ_i ? 1.0 : 0.0

	neuron.membrane_potential += (-neuron.membrane_potential / τ_v + u_i - θ_i * internal_spike) * Δt

	if internal_spike == 1.0 || spike_wagon == 1.0
		push!(neuron.spike_train, neuron.spike_τ)
		neuron.last_spike = current_time
	else
		push!(neuron.spike_train, 0)
	end

	neuron.membrane_potential = clamp(neuron.membrane_potential, -4, θ_i+1)
end

function neurons_LIF_update!(neurons::Vector{N}, current_time::Int, input_spike_train::Vector{Float64}, Δt::Float64; v_i_Γ::Float64=3.0) where {N <: AbstractNeuron}
	liquid_spike_train = input_spike_train |> x -> [x; zeros(length(neurons) - length(x))]
	
	for (neuron, spike_wagon) in zip(neurons, liquid_spike_train)
		neuron_LIF_update!(neuron, current_time, spike_wagon, Δt)
	end
end

# mutable struct SpikingNeuron <: AbstractNeuron
# 	spike_τ::Float64
# 	out_synapses::Vector{AbstractSynapse}  # Specify the concrete type for clarity
# 	last_spike::Float64
# 	spike_train::Vector{Int}       # Binary spike train (1 for spike, 0 for no spike)
# end

mutable struct Synapse <: AbstractSynapse
	weight::Float64
	weight_cap::Tuple{Float64, Float64}
	pre_neuron::AbstractNeuron
	post_neuron::AbstractNeuron
	T_pre::Float64
	T_post::Float64
	spike_τ::Float64
	spike_τ_train::Vector{Float64}
	synaptic_filter::Vector{Float64}
end

function initialize_synapses(neurons::Vector{N}; simulation_length::Int=100) where {N <: AbstractNeuron}
	synapses = Vector{Synapse}()
	C_values = Dict(
		"EE" => 0.2,
		"EI" => 0.1,
		"II" => 0.3,
		"IE" => 0.05
	)

	max_syn_w = 3.0

	# mu = 3  # mean
	# sigma = 8.0  # standard deviation
	# dist = Normal(mu, sigma)
	# weights = rand(dist, length(neurons))
	# weights = abs.(weights .- mu)
	# weights = shuffle(weights ./ maximum(weights) * max_syn_w)

	dist = Normal(0.0, 2.0)
	weights = rand(dist, 1000)
	weights = (-1 .* abs.(weights)) .+ max_syn_w
	weights = shuffle(weights ./ maximum(weights) * max_syn_w)

	i = 1:45
	synaptic_filter = [delayed_synaptic_filter(t, 5, 2) for t in i]
	
	for pre_neuron in neurons
		for post_neuron in neurons
			if pre_neuron !== post_neuron
				# Determine the type of connection (EE, EI, II, IE)
				connection_type = (pre_neuron.spike_τ > 0 ? "E" : "I") * (post_neuron.spike_τ > 0 ? "E" : "I")
				C = C_values[connection_type]
				distance = euclidean_distance(pre_neuron.position, post_neuron.position)

				# Check if synapse should be formed based on connection probability
				if rand() < connection_probability(distance, C)
					synapse = Synapse(
						rand(weights), # rand(0:0.1:max_syn_w),  # Initial random weight
						(0.0, max_syn_w),  # Weight cap
						pre_neuron,  # Pre-synaptic neuron
						post_neuron,  # Post-synaptic neuron
						0.0,  # T_pre initial value
						0.0,  # T_post initial value
						0,  # Spike amplitude
						zeros(Float64, simulation_length),
						synaptic_filter
					)
					push!(synapses, synapse)
					push!(pre_neuron.out_synapses, synapse)
					push!(post_neuron.in_synapses, synapse)
				end
			end
		end
	end
	
	return synapses
end

# STDP Synapse update function
function synapse_STDP_update!(synapse::Synapse, current_time::Int, Δt::Float64; syn_filter_t::Int=90)
	synapse.spike_τ_train = conv(synapse.synaptic_filter, synapse.pre_neuron.spike_train[max(current_time-syn_filter_t+1,1):current_time])
	synapse.spike_τ = synapse.weight * synapse.spike_τ_train[current_time]

	astro_activity = []
	for astrocyte in synapse.pre_neuron.linked_astrocytes
		push!(astro_activity, astrocyte.A_astro)
	end
	for astrocyte in synapse.post_neuron.linked_astrocytes
		push!(astro_activity, astrocyte.A_astro)
	end

	if astro_activity == []
		A_minus = 0.15
	else
		A_minus = mean(astro_activity)
	end
	
	A_plus = 0.15
		
	τ_plus = 10.0  # ms
	τ_minus = 10.0  # ms
	a_plus = 0.1
	a_minus = 0.1

	# # Update traces based on the spike train
	# T_pre_decay = exp(-Δt / τ_plus)
	# T_post_decay = exp(-Δt / τ_minus)

	# synapse.T_pre = synapse.T_pre * T_pre_decay + a_plus * synapse.pre_neuron.spike_train[current_time]
	# synapse.T_post = synapse.T_post * T_post_decay + a_minus * synapse.post_neuron.spike_train[current_time]

	# Update traces based on the spike train
	synapse.T_pre += (-synapse.T_pre + a_plus * synapse.pre_neuron.spike_train[current_time]) * (Δt / τ_plus)
	synapse.T_post += (-synapse.T_post + a_minus * synapse.post_neuron.spike_train[current_time]) * (Δt / τ_minus)

	# STDP weight update based on the last spike times

	# Potentiation due to pre-synaptic spike
	synapse.weight += A_plus * synapse.T_pre * abs.(synapse.post_neuron.spike_train[current_time]) * Δt
	# Depression due to post-synaptic spike
	synapse.weight -= A_minus * synapse.T_post * abs.(synapse.pre_neuron.spike_train[current_time]) * Δt

	# Clamp weight within reasonable bounds
	synapse.weight = clamp(synapse.weight, synapse.weight_cap[1], synapse.weight_cap[2])
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
	liquid_neurons::Vector{AbstractNeuron}
	astro_liq_t::Int
end
function initialize_astrocytes(num_astrocytes::Int, liquid_neurons::Vector{S}; astro_liq_t::Int=1) where {S <: AbstractNeuron}
	astrocytes = Vector{Astrocyte}()
	for _ in 1:num_astrocytes
		modulated_neurons = unique(rand(liquid_neurons, 750))
		# modulated_neurons = liquid_neurons

		astrocyte = Astrocyte(
			0.15,      	# Initial A_astro value
			1.0,     # τ_astro should be set according to your model specifics
			0.01,     # w_astro, the weight for the astrocyte's influence
			0.0,      	# b_astro, bias or base level of astrocyte's activity
			1.0, 		# Γ_astro, the gain for the astrocyte's influence
			modulated_neurons,
			astro_liq_t
		)
		push!(astrocytes, astrocyte)
		for n in modulated_neurons
			push!(n.linked_astrocytes, astrocyte)
		end
	end
	return astrocytes
end

# Astrocyte LIM model update function
function astrocyte_LIM_update!(astrocyte::Astrocyte, current_time::Int, input_spike_train::Matrix{Float64}, Δt::Float64)
	# Calculate the total spikes from liquid and input neurons at the current time
	# liquid_spikes = sum(synapse.pre_neuron.spike_train[current_time] for synapse in astrocyte.liquid_synapses)
	# input_spikes = sum(u_i)

	input_spikes = sum(input_spike_train)/size(input_spike_train)[2]
	liquid_spikes = sum(hcat([abs.(n.spike_train[max(current_time-astrocyte.astro_liq_t+1, 1):current_time]) for n in astrocyte.liquid_neurons]...))[1]/astrocyte.astro_liq_t
	# liquid_spikes = sum(abs.(n.spike_train[current_time]) for n in astrocyte.liquid_neurons)

	# fetch last n times from u_i & liquid neurons
	# average over time => 1. less jumpy-ness

	# Compute the change in astrocyte activity
	dA_astro_dt = (-astrocyte.A_astro * astrocyte.Γ_astro + astrocyte.w_astro * (liquid_spikes - input_spikes) + astrocyte.b_astro) / astrocyte.τ_astro
	
	# Update the astrocyte's state using Euler integration
	astrocyte.A_astro += dA_astro_dt * Δt
end

# LIM update function for all astrocytes
function astrocytes_LIM_update!(astrocytes::Vector{Astrocyte}, current_time::Int, input_spike_trains::Matrix{Float64}, Δt::Float64)
	s_i = []
	s_l = []
	println("Mean of A_astro before: ", mean([astrocyte.A_astro for astrocyte in astrocytes]))
	for astrocyte in astrocytes
		# momentary approximation of input & liquid spikes
		input_spikes = sum(input_spike_trains)
		liquid_spikes = sum(hcat([abs.(n.spike_train[max(current_time-astrocyte.astro_liq_t+1, 1):current_time]) for n in astrocyte.liquid_neurons]...))[1]/astrocyte.astro_liq_t
		push!(s_i, input_spikes)
		push!(s_l, liquid_spikes)

		astrocyte_LIM_update!(astrocyte, current_time, input_spike_trains, Δt)
	end
	println("Mean of A_astro after: ", mean([astrocyte.A_astro for astrocyte in astrocytes]))

	println("Astrocyte activity; mean liq-in spike ratio (over last $(size(input_spike_trains)[2])): ", mean(s_l ./ s_i))
	println("Mean liquid spikes: ", mean(s_l))
	println("Mean input spikes: ", mean(s_i))
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