using Pkg

Pkg.activate("./LSM.jl")

# TODO: 
# 	Technical:
#	- 1. Add input adapter, spiking neurons, ouptut point neuron, & inhib synapses
#	- 2. Connect spiking neurons (input from input adapter) with synapses to liquid neurons 
#	- 3. Connect liquid neurons with synapses to readout neurons
# 	- 4. Connect astrocytes to local neurons and synapses (in pocket of local neurons & connect to their synapses)
# 	- 5. Feed spike to one side of 3D neural system
# 	Practical:
# 	- 0. Isolate mutable parts from immutable structures (speed up => simpler job for parallelism)
# 	- 1. Add multi-processing
#   - 2. Reduce precision of floating point numbers (speed up)

include("graphics.jl")
include("lsm.jl")

@time lsm = LiquidStateMachine(grid_type="cube")
@time simulate_w_hist!(lsm, spike_train_generator=lsm.stim_spike_train)
@time simulate_w_hist!(lsm)

for _ in 1:3
	@time simulate_w_hist!(lsm, spike_train_generator=lsm.stim_spike_train)
end
for _ in 1:2
	@time simulate_w_hist!(lsm)
end

@time create_plots(
	"EXP-006-",
	lsm.reservoir_hist["neuron_membrane_hist"], 
	lsm.reservoir_hist["synapse_weight_hist"], 
	lsm.reservoir_hist["astrocyte_A_hist"]
)

println("DONE!")

reset_hist!(lsm)