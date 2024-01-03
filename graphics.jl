using Plots
using PlotThemes

gr()
theme(:juno)

function create_plots(neuron_states, synapse_states, astrocyte_A_states)
    duration = size(neuron_states, 2)
    anim = @animate for t in 1:duration
        p1 = plot(neuron_states[:, t], title = "Neuron States at t=$(t)(ms)", legend = false, color=:lightgreen, ylims = (-3, 8))
        hline!(p1, [5], color=:red, linestyle=:dash)
        p2 = plot(synapse_states[:, t], title = "Synapse States at t=$(t)(ms)", legend = false, color=:orange, ylims = (0, 4))
        hline!(p2, [3.0], color=:lightblue, linestyle=:dash)
        # hline!(p2, [-3.0], color=:lightblue, linestyle=:dash)
        p3 = plot(astrocyte_A_states[:, t], title = "Astrocyte States at t=$(t)(ms)", legend = false, color=:purple, ylims = (-1, 0.5))
        hline!(p3, [0.15], color=:yellow)
        plot(p1, p2, p3, layout = (3, 1), size = (900, 900))
    end
    gif(anim, "lsm_states.gif", fps = 10)
end