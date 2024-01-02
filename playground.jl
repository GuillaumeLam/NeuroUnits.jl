using Pkg

Pkg.activate("./LSM.jl")

using Plots
include("lsm.jl")

# Initialize components
@time lsm = LiquidStateMachine()

input_current = [Float64(rand()>0.7) for _ in 1:10]

# Simulate the reservoir for the given number of timesteps
@time neuron_states, synapse_states, astrocyte_states = simulate_hist!(lsm, input_current)

function create_plots(neuron_states, synapse_states, astrocyte_states)
    duration = size(neuron_states, 2)
    anim = @animate for t in 1:duration
        p1 = plot(neuron_states[:, t], title = "Neuron States at t=$t", legend = false, color=:lightgreen, ylims = (-1, 3))
        p2 = plot(synapse_states[:, t], title = "Synapse States at t=$t", legend = false, color=:orange, ylims = (-1, 1))
        p3 = plot(astrocyte_states[:, t], title = "Astrocyte States at t=$t", legend = false, color=:purple, ylims = (0, 0.5))
        plot(p1, p2, p3, layout = (3, 1))
    end
    gif(anim, "lsm_states.gif", fps = 10)
end

# Call the function with the states and duration
@time create_plots(neuron_states, synapse_states, astrocyte_states)