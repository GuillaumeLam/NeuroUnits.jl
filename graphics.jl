using Plots
using PlotThemes

gr()
theme(:juno)

function create_plots(specific_title, neuron_states, synapse_states, astrocyte_A_states)
    duration = size(neuron_states, 2)
    anim = @animate for t in 1:duration

        p1 = plot(neuron_states[:, t], title = "Neuron States at t=$(t)(ms)", legend = false, color=:lightgreen, ylims = (-5, 22), xlabel = "Neuron Index", ylabel = "Membrane Potential")
        hline!(p1, [20], color=:red, linestyle=:dash)

        sorted_synapse_states = sort(synapse_states[:, t])
        max_syn_w = 7.0

        p2 = plot(sorted_synapse_states, title = "Ordered Synapse States at t=$(t)(ms)", legend = false, color=:orange, ylims = (-(max_syn_w+0.5), max_syn_w+0.5), xlabel = "Synapse Index", ylabel = "Weight")
        hline!(p2, [max_syn_w], color=:lightblue, linestyle=:dash)
        # hline!(p2, [-3.0], color=:lightblue, linestyle=:dash)

        p3 = plot(astrocyte_A_states[:, t], title = "Astrocyte States at t=$(t)(ms)", legend = false, color=:purple, ylims = (-2.5, 2.5), xlabel = "Astrocyte Index", ylabel = "Modulation")
        hline!(p3, [0.15], color=:yellow)
        
        plot(p1, p2, p3, layout = (3, 1), size = (900, 900))
    end
    gif(anim, "LSM.jl/out/lsm_states_$specific_title.gif", fps = 10)
end