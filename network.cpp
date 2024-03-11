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
};
