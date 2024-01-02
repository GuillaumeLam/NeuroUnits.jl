using Plots
using PlotThemes

gr()
theme(:juno)

function create_plots(neuron_states, synapse_states, astrocyte_states)
    duration = size(neuron_states, 2)
    anim = @animate for t in 1:duration
        p1 = plot(neuron_states[:, t], title = "Neuron States at t=$t", legend = false, color=:lightgreen, ylims = (-3, 4))
        p2 = plot(synapse_states[:, t], title = "Synapse States at t=$t", legend = false, color=:orange, ylims = (-0.7, 1.3))
        p3 = plot(astrocyte_states[:, t], title = "Astrocyte States at t=$t", legend = false, color=:purple, ylims = (-1, 7))
        plot(p1, p2, p3, layout = (3, 1))
    end
    gif(anim, "lsm_states.gif", fps = 10)
end