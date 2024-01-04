# function sinusoidal_input_factory(num_neurons::Int, frequency::Float64, simulation_length::Int, amplitude::Float64 = 1.0)
#     ω = 2 * π * frequency  # Angular frequency
#     function u_i_f(t)
#         return [Float64(amplitude * sin(ω * t)>=0.95) for _ in 1:num_neurons]
#     end
#     return u_i_f
# end

# # Function to adapt a dataset for the reservoir
# function adapt_dataset(dataset, reservoir_neurons)
#     # Logic to adapt dataset and determine which neurons receive input
#     # This could involve normalizing the dataset, mapping data points to neurons, etc.
#     adapted_data = []  # Placeholder for adapted data
#     for data_point in dataset
#         # Process each data point
#         # Placeholder: Randomly assign data to neurons for illustration
#         push!(adapted_data, (data_point, rand(reservoir_neurons)))
#     end
#     return adapted_data
# end
