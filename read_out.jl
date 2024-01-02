module ReadOut

export Perceptron, update_perceptron, activate

struct Perceptron
    weights::Array{Float64}
    bias::Float64
    activation_function::Function
end

# Activation function for the perceptron
function activate(p::Perceptron, input::Float64)
    return p.activation_function(dot(p.weights, input) + p.bias)
end

# Function to update the perceptron
function update_perceptron(perceptron::Perceptron, inputs, targets, learning_rate::Float64)
    for (input, target) in zip(inputs, targets)
        prediction = activate(perceptron, input)
        error = target - prediction
        # Update weights and bias based on the error and learning rate
        perceptron.weights .+= learning_rate * error * input
        perceptron.bias += learning_rate * error
    end
end

end
