using Pkg

Pkg.activate("./LSM.jl")

include("reservoir.jl")
using Plots

# Initialize components
neurons = initialize_neurons(10)        # Initialize 10 neurons
synapses = initialize_synapses(neurons, 30)  # Initialize 20 synapses
astrocytes = initialize_astrocytes(neurons, 2) # Initialize 2 astrocytes

# Number of timesteps to simulate
num_timesteps = 100
input_current = [Float64(rand()>0.7) for _ in 1:10]

# Simulate the reservoir for the given number of timesteps
@time neuron_states, synapse_states, astrocyte_states = simulate_hist!(input_current, num_timesteps, neurons, synapses, astrocytes)

function create_plots(neuron_states, synapse_states, astrocyte_states, duration)
    anim = @animate for t in 1:duration
        p1 = plot(neuron_states[:, t], title = "Neuron States at t=$t", legend = false, color=:lightgreen)
        p2 = plot(synapse_states[:, t], title = "Synapse States at t=$t", legend = false, color=:orange)
        p3 = plot(astrocyte_states[:, t], title = "Astrocyte States at t=$t", legend = false, color=:purple)
        plot(p1, p2, p3, layout = (3, 1))
    end
    gif(anim, "lsm_states.gif", fps = 10)
end

# Call the function with the states and duration
create_plots(neuron_states, synapse_states, astrocyte_states, num_timesteps)