#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstddef>
#include <Eigen/Dense>

using std::cout; using std::cin; using std::endl;
using std::string;
using std::vector;
using std::size_t;

using Eigen::VectorXd; using Eigen::MatrixXd;

/**
 * @brief Abstract base class for activation functions.
 *
 * Base for objects representing activation functions in the context of a neural
 * network.
 *
 * Neural networks are structures resulting in predictive models. Given a set of
 * numerical inputs, such as the brightness of pixels in a grayscale image, we
 * seek to accurately predict an output which is based on patterns to be found
 * in the output, such as the handwritten digit represented by the image.
 *
 * A neural network is composed of layers of neurons. Neurons are connected to
 * all neurons in the layers before and after the one they belong to. This
 * connection is represented by a weight. The value of a neuron is calculated by
 * summing the values of all neurons connected to it in the previous layer,
 * multiplied by their assigned weight, and then adding a bias. The goal of a
 * neural network is thus to transform the input into output, adjusting its
 * weights and biases to accurately represent the patterns present in the data.
 *
 * If we were to stop there, we would have a linear model. Applying an
 * activation function common to all neurons in a layer on top of the sum, we
 * introduce non-linearity to the model. With it, the network can learn patterns
 * that are non-linear, and thus the network is much more versatile.
 */
template<class Implementation>
struct Activation {
    /**
     * @brief Activation function.
     *
     * Calculate the activation of a given set of weighted inputs.
     *
     * @param weightedInputs Vector of the weighted inputs of a neural
     *                       network layer.
     *
     * @return Activations of the given layer.
     */
    static VectorXd activate(VectorXd weightedInputs) {
        return Implementation::activateInternal(weightedInputs);
    };

    /**
     * @brief Gradient of activations.
     *
     * Calculate the gradient of a layer's activations over its weighted
     * inputs.
     *
     * @param weightedInputs Vector of the weighted inputs of a neural
     *                       network layer.
     *
     * @return Gradient of a layer's activations over its weighted inputs.
     */
    static VectorXd gradient(VectorXd weightedInputs) {
        return Implementation::gradientInternal(weightedInputs);
    };

};

/**
 * @brief Identity activation function.
 *
 * Vectorial identity. This is for use for demonstration in the context of a
 * neural network, to see what the behavior of the network would be like without
 * using an activation function.
 * Activation functions introduce non-linearity to the model, which allows it to
 * learn more complex patterns compared to a linear model. Therefore, using this
 * activation function should lead to worse results.
 */
struct Identity: Activation<Identity> {
    static VectorXd activateInternal(VectorXd weightedInputs) {
        return weightedInputs;
    }

    static VectorXd gradientInternal(VectorXd weightedInputs) {
        return VectorXd::Ones(weightedInputs.size());
    }
};

/**
 * @brief Sigmoid activation function.
 *
 * Vectorial implementation of the sigmoid function.
 * In the context of a neural network, it is widely used as an activation
 * function. Being essentially a "smoothed-out" step function, it is a direct
 * step up from perceptron networks, allowing gradual improvements and more
 * nuance over a strictly binary structure.
 * When paired with a cross-entropy cost function, it allows for a
 * simplified calculation of the network's error which results in faster
 * learning and faster computing.
 */
struct Sigmoid: Activation<Sigmoid> {
    static VectorXd activateInternal(VectorXd weightedInputs) {
        return 1 / (1 + (-weightedInputs).array().exp());
    }

    static VectorXd gradientInternal(VectorXd weightedInputs) {
        VectorXd sigmoidOfWeightedInputs =
            Sigmoid::activate(weightedInputs);

        return sigmoidOfWeightedInputs.array() * (
            VectorXd::Ones(sigmoidOfWeightedInputs.size())
            - sigmoidOfWeightedInputs
        ).array();
    }
};

/**
 * @brief Configurable neural network.
 *
 * Application of Michael Nielsen's four equations of back-propagation
 * following a neural network structure.
 *
 * @tparam ActivationFunction Activation to be used throughought the network.
 *
 * @param layerSizes Array containing the sizes of each layer to be
 *                   constructed in the network.
 */
template<class ActivationFunction>
class Network {
public:
    vector<size_t> layerSizes;
    size_t amountOfLayers;
    vector<MatrixXd> weightsByLayer;
    vector<VectorXd> biasesByLayer;

    Network(
        vector<size_t> ls
    ):
        layerSizes(ls),
        amountOfLayers(ls.size())
    {
        for (
            size_t layerIndex = 0;
            layerIndex < amountOfLayers - 1;
            ++layerIndex
        ) {
            size_t outgoingLayerSize = layerSizes[layerIndex];
            size_t incomingLayerSize = layerSizes[layerIndex + 1];

            MatrixXd layerWeights = MatrixXd::Random(
                incomingLayerSize, outgoingLayerSize
            );
            VectorXd layerBiases = VectorXd::Random(
                incomingLayerSize
            );

            /** TODO: Include optional division by square root of amount of \
             *        outgoing weights.
             */

            weightsByLayer.push_back(layerWeights);
            biasesByLayer.push_back(layerBiases);
        }
    }

    /**
     * @brief Feed input values through the network. Cache every step.
     *
     * Pass a given array of input activations through the network, caching and
     * returning every intermediate result.
     *
     * @param inputLayer Array of input activations to be fed through the \
     *                   network.
     *
     * @return Tuple composed of weighted inputs and activations per layer.
     */
    std::tuple<vector<VectorXd>,vector<VectorXd>>
        feedforward(VectorXd inputLayer)
    {
        VectorXd layerActivations = inputLayer;
        vector<VectorXd> weightedInputsByLayer;
        vector<VectorXd> activationsByLayer = {layerActivations};

        for (
            size_t layerIndex = 0;
            layerIndex < amountOfLayers - 1;
            ++layerIndex
        ) {
            MatrixXd layerWeights = weightsByLayer[layerIndex];
            VectorXd layerBiases = biasesByLayer[layerIndex];

            VectorXd layerWeightedInputs =
                layerWeights * layerActivations + layerBiases;
            layerActivations = ActivationFunction::activate(-layerWeightedInputs);
            weightedInputsByLayer.push_back(layerWeightedInputs);
            activationsByLayer.push_back(layerActivations);
        }

        return std::make_tuple(weightedInputsByLayer, activationsByLayer);
    }

    /**
     * @brief Get network output given input.
     *
     * Pass a given array of input activations through the network and return
     * the corresponding output layer.
     * Same method as Network::feedforward, only without caching intermediate
     * steps and solely returning the output layer's activations.
     *
     * @param inputLayer Array of input activations to be fed through the \
     *                   network.
     *
     * @return Output layer activations.
     */
    VectorXd feedforward_without_caching(VectorXd inputLayer)
    {
        VectorXd layerActivations = inputLayer;

        for (
            size_t layerIndex = 0;
            layerIndex < amountOfLayers - 1;
            ++layerIndex
        ) {
            MatrixXd layerWeights = weightsByLayer[layerIndex];
            VectorXd layerBiases = biasesByLayer[layerIndex];

            VectorXd layerWeightedInputs =
                layerWeights * layerActivations + layerBiases;
            layerActivations = ActivationFunction::activate(-layerWeightedInputs);
        }
        return layerActivations;
    }
};
