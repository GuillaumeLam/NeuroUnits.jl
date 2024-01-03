using Pkg

Pkg.activate("./LSM.jl")

coin(p0=0.1) = rand()>p0 ? 1.0 : 0.0

function factory(p0, n, strength=20)
	function u_i_f(t)
		[coin(p0)*strength for _ in 1:n]
	end
	return u_i_f
end

# function sinusoidal_input_factory(num_neurons::Int, frequency::Float64, simulation_length::Int, amplitude::Float64 = 1.0)
#     ω = 2 * π * frequency  # Angular frequency
#     function u_i_f(t)
#         return [Float64(amplitude * sin(ω * t)>=0.95) for _ in 1:num_neurons]
#     end
#     return u_i_f
# end

include("reservoir-components.jl")

# Initialize the LSM components
num_liq_neurons = 1000
num_liq_synapses = 3000
num_liq_astrocytes = 20
num_spk_neurons = 85

# freq = 10

reservoir_hist = Dict(
	"neuron_membrane_hist" => Matrix{Float64}(undef, num_liq_neurons, 0),
	"synapse_weight_hist" => Matrix{Float64}(undef, num_liq_synapses, 0),
	"astrocyte_A_hist" => Matrix{Float64}(undef, num_liq_astrocytes, 0),
)

# Initialize the neurons, synapses, and astrocytes
liq_neurons = initialize_neurons(num_liq_neurons)
liq_synapses = initialize_synapses(num_liq_synapses, liq_neurons)
liq_astrocytes = initialize_astrocytes(num_liq_astrocytes, liq_neurons)

u_i_t_stim = factory(0.05, num_spk_neurons)
u_i_t_rest = factory(0.95, num_spk_neurons)

@time simulate_w_hist!(reservoir_hist, u_i_t_stim, liq_neurons, liq_synapses, liq_astrocytes)
@time simulate_w_hist!(reservoir_hist, u_i_t_rest, liq_neurons, liq_synapses, liq_astrocytes)

include("graphics.jl")

@time create_plots(
	reservoir_hist["neuron_membrane_hist"], 
	reservoir_hist["synapse_weight_hist"], 
	reservoir_hist["astrocyte_A_hist"]
)

println("DONE!!!")