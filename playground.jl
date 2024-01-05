using Pkg

Pkg.activate("./LSM.jl")

# TODO: 
# 	Technical:
#	- 1. Add input adapter, spiking neurons, ouptut point neuron, & inhib neurons/synapses
#	- 2. Connect spiking neurons (input from input adapter) with synapses to liquid neurons
#	- 3. Connect liquid neurons with synapses to readout neurons
# 	Practical:
# 	- 0. Isolate mutable parts from immutable structures (speed up => simpler job for parallelism)
# 	- 1. Add multi-processing

include("graphics.jl")
include("lsm.jl")

@time lsm = LiquidStateMachine(grid_type="cube")
@time simulate_w_hist!(lsm, u_i_f=lsm.u_i_t_stim)
@time simulate_w_hist!(lsm)

for _ in 1:3
	@time simulate_w_hist!(lsm, u_i_f=lsm.u_i_t_stim)
end

@time create_plots(
	"EXP-002-num_spk_neurons=80-max_syn_w=5-5-rest_p=0.1",
	lsm.reservoir_hist["neuron_membrane_hist"], 
	lsm.reservoir_hist["synapse_weight_hist"], 
	lsm.reservoir_hist["astrocyte_A_hist"]
)

println("DONE!")

reset_hist!(lsm)