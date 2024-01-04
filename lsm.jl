include("reservoir-components.jl")
include("read_in.jl")
# include("read_out.jl")

struct LiquidStateMachine
    # spk_neurons::Vector{AbstractNeuron}
    # readin_synapses::Vector{Synapse}
    liq_neurons::Vector{AbstractNeuron}
    liq_synapses::Array{Synapse}
    liq_astrocytes::Array{Astrocyte}
    # readout_synapses::Vector{Synapse}
    # readout_neurons::Vector{AbstractNeuron}

    reservoir_hist::Dict{String, Matrix{Float64}}

    u_i_t_stim::Function
    u_i_t_rest::Function

    function LiquidStateMachine(;num_spk_neurons::Int=150, num_liq_neurons::Int=1000, num_liq_synapses::Int=2500, num_liq_astrocytes::Int=1500, signal_gain::Float64=7.0)
        # freq = 10

        u_i_t_stim = factory(0.95, num_spk_neurons, signal_gain)
        u_i_t_rest = factory(0.05, num_spk_neurons, signal_gain)

        reservoir_hist = Dict(
            "neuron_membrane_hist" => Matrix{Float64}(undef, num_liq_neurons, 0),
            "synapse_weight_hist" => Matrix{Float64}(undef, num_liq_synapses, 0),
            "astrocyte_A_hist" => Matrix{Float64}(undef, num_liq_astrocytes, 0),
        )

        # Initialize the neurons, synapses, and astrocytes
        liq_neurons = initialize_neurons(num_liq_neurons)
        liq_synapses = initialize_synapses(num_liq_synapses, liq_neurons)
        liq_astrocytes = initialize_astrocytes(num_liq_astrocytes, liq_synapses)

        new(
            liq_neurons, 
            liq_synapses, 
            liq_astrocytes, 
            reservoir_hist,
            u_i_t_stim,
            u_i_t_rest
        )
    end
end

function simulate!(lsm::LiquidStateMachine; u_i_f=nothing, duration::Int=100, Δt::Float64=1.0)
	for current_time in 1:duration
		
        if isnothing(u_i_f)
            u_i = lsm.u_i_t_rest(current_time)
        else
            u_i = u_i_f(current_time)
        end
        # when read_in is implemented, u_i_f => u_i will be the stimulus passed to readin

		neurons_LIF_update!(lsm.liq_neurons, current_time, u_i, Δt)
		synapses_STDP_update!(lsm.liq_synapses, current_time, Δt)
		astrocytes_LIM_update!(lsm.liq_astrocytes, current_time, u_i, Δt)
	end
end

function simulate_w_hist!(lsm::LiquidStateMachine; u_i_f=nothing, duration::Int=100, Δt::Float64=1.0)
	neuron_membrane_hist = Matrix{Float64}(undef, length(lsm.liq_neurons), duration)
	synapse_weight_hist = Matrix{Float64}(undef, length(lsm.liq_synapses), duration)
	astrocyte_A_hist = Matrix{Float64}(undef, length(lsm.liq_astrocytes), duration)

	for current_time in 1:duration
		
		println("current_time: ", current_time)

		if isnothing(u_i_f)
            u_i = lsm.u_i_t_rest(current_time)
        else
            u_i = u_i_f(current_time)
        end

		neurons_LIF_update!(lsm.liq_neurons, current_time, u_i, Δt)
		synapses_STDP_update!(lsm.liq_synapses, current_time, Δt)
		astrocytes_LIM_update!(lsm.liq_astrocytes, current_time, u_i, Δt)
	
		# Record neuron membrane potentials
		for (i, neuron) in enumerate(lsm.liq_neurons)
			neuron_membrane_hist[i, current_time] = neuron.membrane_potential
		end
		# Record synapse weights
		for (i, synapse) in enumerate(lsm.liq_synapses)
			synapse_weight_hist[i, current_time] = synapse.weight
		end
		# Record astrocyte A_astro
		for (i, astrocyte) in enumerate(lsm.liq_astrocytes)
			astrocyte_A_hist[i, current_time] = astrocyte.A_astro
		end
	end

	lsm.reservoir_hist["neuron_membrane_hist"] = hcat(lsm.reservoir_hist["neuron_membrane_hist"], neuron_membrane_hist)
	lsm.reservoir_hist["synapse_weight_hist"] = hcat(lsm.reservoir_hist["synapse_weight_hist"], synapse_weight_hist)
	lsm.reservoir_hist["astrocyte_A_hist"] = hcat(lsm.reservoir_hist["astrocyte_A_hist"], astrocyte_A_hist)

    return lsm.reservoir_hist
end

function (lsm::LiquidStateMachine)(input::Vector{Float64})

    # h1 = lsm.read_in_connections(input)

    h2 = simulate!(lsm, input)

    # output = lsm.read_out_connections(h2)

    return h2
end

function Base.show(io::IO, ::MIME"text/plain", a::LiquidStateMachine)
    println(io, "Liquid State Machine assembled!!!!")
end