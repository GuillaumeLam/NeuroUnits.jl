#########################
# NEURON GRID FUNCTIONS #
#########################

function generate_cubic_grid(grid_size::Tuple{Int, Int, Int}, spacing::Float64)
    cube_grid_positions = Vector{Tuple{Float64, Float64, Float64}}()
    x_dim, y_dim, z_dim = grid_size
    for x in 1:x_dim
        for y in 1:y_dim
            for z in 1:z_dim
                push!(cube_grid_positions, (x * spacing, y * spacing, z * spacing))
            end
        end
    end
    return cube_grid_positions
end

# Example usage
# cubic_grid_positions = generate_cubic_grid((10, 10, 10), 1.0)  # 10x10x10 grid with 1.0 unit spacing

function generate_hexagonal_prism_grid(radius::Int, layers::Int, spacing::Float64)
    # Generate the hexagonal pattern in the XY plane
    hex_points = []
    for q in -radius:radius
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in r1:r2
            push!(hex_points, (q, r))
        end
    end
    
    # Convert axial coordinates to 3D cartesian coordinates and extrude in the Z dimension
    hex_grid_positions = Vector{Tuple{Float64, Float64, Float64}}()
    for (q, r) in hex_points
        x = spacing * (sqrt(3) * q + sqrt(3)/2 * r)
        y = spacing * (3/2 * r)
        for z in 1:layers
            push!(hex_grid_positions, (x, y, z * spacing))
        end
    end
    
    return hex_grid_positions
end

function tile_hexagonal_prism_grid(base_grid_positions::Vector{Tuple{Float64, Float64, Float64}}, num_tiles::Tuple{Int, Int}, spacing::Float64)
    tiled_positions = Vector{Tuple{Float64, Float64, Float64}}()
    base_width = sqrt(3) * spacing  # Width of a single hexagon calculated from its height (spacing)
    
    for tile_x in 0:num_tiles[1]-1
        for tile_y in 0:num_tiles[2]-1
            # Calculate the offset for each tile
            offset_x = tile_x * base_width * 3 / 2  # 3/2 comes from hexagon tiling geometry
            offset_y = tile_y * spacing * sqrt(3)  # sqrt(3) comes from hexagon tiling geometry
            
            # Apply the offset to each point in the base grid
            for position in base_grid_positions
                push!(tiled_positions, (position[1] + offset_x, position[2] + offset_y, position[3]))
            end
        end
    end
    
    return tiled_positions
end

# Example usage: Create a base grid and then tile it
# base_hex_grid_positions = generate_hexagonal_prism_grid(3, 1, 1.0)  # Single-layer hex grid
# tiled_hex_grid_positions = tile_hexagonal_prism_grid(base_hex_grid_positions, (3, 3), 1.0)  # 3x3 tiling of the hex grid

# function generate_triangular_prism_grid(side_length::Int, layers::Int, spacing::Float64)
#     # Generate the triangular pattern in the XY plane
#     triangle_points = []
#     for x in 0:side_length-1
#         for y in 0:(side_length - 1 - x)
#             push!(triangle_points, (x, y))
#             if x != 0  # Add mirrored point for the other half of the triangle
#                 push!(triangle_points, (-x, y))
#             end
#         end
#     end
    
#     # Convert 2D grid to 3D points and extrude in the Z dimension
#     tri_grid_positions = Vector{Tuple{Float64, Float64, Float64}}()
#     for (x, y) in triangle_points
#         # Offset each row to create a staggered arrangement
#         offset = (x % 2) * (spacing / 2)
#         for z in 1:layers
#             push!(tri_grid_positions, (x * spacing, y * spacing + offset, z * spacing))
#         end
#     end
    
#     return tri_grid_positions
# end

# function tile_triangular_prism_grid(base_grid_positions::Vector{Tuple{Float64, Float64, Float64}}, num_tiles::Int, spacing::Float64)
#     tiled_positions = Vector{Tuple{Float64, Float64, Float64}}()
#     base_height = spacing * sqrt(3) / 2  # Height of an equilateral triangle
    
#     for tile_x in 0:num_tiles-1
#         for tile_y in 0:num_tiles-1-tile_x  # Ensure we tile in a triangular fashion
#             # Calculate the offset for each tile
#             offset_x = tile_x * spacing
#             offset_y = tile_y * base_height
            
#             # Apply the offset to each point in the base grid
#             for position in base_grid_positions
#                 push!(tiled_positions, (position[1] + offset_x, position[2] + offset_y, position[3]))
#             end
#         end
#     end
    
#     return tiled_positions
# end

# # Example usage: Create a single base grid and then tile it
# # base_tri_grid_positions = generate_triangular_prism_grid(10, 1, 1.0)  # Single-layer triangle grid
# # tiled_tri_grid_positions = tile_triangular_prism_grid(base_tri_grid_positions, 3, 1.0)  # 3 tiles along one side

# function generate_truncated_octahedron_grid(grid_size::Tuple{Int, Int, Int}, spacing::Float64)
#     x_dim, y_dim, z_dim = grid_size
#     # Create a list to hold the grid points
#     grid_points = []

#     # Generate the centers of the truncated octahedra
#     for x in 1:x_dim
#         for y in 1:y_dim
#             for z in 1:z_dim
#                 # Calculate the center point for the current truncated octahedron
#                 center = (x * spacing, y * spacing, z * spacing)
#                 push!(grid_points, center)
#             end
#         end
#     end
#     return grid_points
# end

# # Example usage: Create a grid of truncated octahedron centers
# # grid_size = (3, 3, 3)  # Define the size of the grid
# # spacing = 2.0          # Define the spacing between the centers
# # truncated_octahedron_centers = generate_truncated_octahedron_grid(grid_size, spacing)

################################
# SYNAPSE CONNECTION FUNCTIONS #
################################

# Euclidean distance calculation
function euclidean_distance(pos1::Tuple{Float64, Float64, Float64}, pos2::Tuple{Float64, Float64, Float64})
    return sqrt(sum((p1 - p2)^2 for (p1, p2) in zip(pos1, pos2)))
end

# Connection probability function
function connection_probability(distance::Float64, C::Float64, lambda::Float64=3.0)
    return C * exp(-(distance / lambda)^2)
end

function alpha_synaptic_filter(t, τ)
    return (t > 0) ? (t / τ) * exp(1 - t / τ) : 0
end

# Input Variables
# t: (ms) strength of signal given t 
# Context Variables
# τ: (ms) time of peak synaptic response
# delay: (ms) delay of synaptic response
function delayed_synaptic_filter(t, τ, delay)
    return (t > delay) ? (t - delay) / τ * exp(1 - (t - delay) / τ) : 0.0
end

################################
# STIMULUS GENERATOR FUNCTIONS #
################################

coin(p1=0.5) = rand()<p1 ? 1.0 : 0.0

function coin_factory(p1, n, strength::Float64=1.0)
	function spike_train_generator(t)
		[coin(p1)*strength for _ in 1:n]
	end
	return spike_train_generator
end

function freq_factory(n, strength::Float64=1.0; freq::Int=10)
    # freq is in hertz; (max:1000, min:1)

    function spike_train_generator(t)
        if t % ceil(1000/freq) == 0
            return [strength for _ in 1:n]
        else
            return [0.0 for _ in 1:n]
        end
    end
    return spike_train_generator
end