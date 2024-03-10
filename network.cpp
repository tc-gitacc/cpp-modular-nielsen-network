#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include <Eigen/Dense>

using std::cout; using std::cin; using std::endl;
using std::string;
using std::vector;
using std::size_t;

using Eigen::VectorXd; using Eigen::MatrixXd;

/**
 * @brief Configurable neural network.
 *
 * Application of Michael Nielsen's four equations of back-propagation
 * following a neural network structure.
 *
 * @param layerSizes Array containing the sizes of each layer to be \
 *                   constructed in the network.
 */
class Network {
public:
    vector<size_t> layerSizes;

    Network(
        vector<size_t> ls
    ):
        layerSizes(ls)
    {
        vector<MatrixXd> weightsByLayer;
        size_t amountOfLayers = layerSizes.size();
        for (
            size_t layerPosition = 0;
            layerPosition < amountOfLayers - 1;
            ++layerPosition
        ) {
            size_t outgoingLayerSize = layerSizes[layerPosition];
            size_t incomingLayerSize = layerSizes[layerPosition + 1];

            MatrixXd layerWeights = MatrixXd::Random(
                incomingLayerSize, outgoingLayerSize
            );
            VectorXd layerBiases = VectorXd::Random(
                incomingLayerSize
            );

            layerWeights = 0.5 * (
                layerWeights
                + MatrixXd::Ones(
                    incomingLayerSize, outgoingLayerSize
                )
            );
            // TODO: Include optional division by square root of amount of outgoing weights
            layerBiases = 0.5 * (
                layerBiases + VectorXd::Ones(incomingLayerSize)
            );

            weightsByLayer.push_back(layerWeights);
        }
    }
};
