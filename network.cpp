#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using std::cout; using std::cin; using std::endl;
using std::string;
using std::vector;

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
    vector<int> layerSizes;

    Network(
        vector<int> ls
    ):
        layerSizes(ls)
    {
        vector<MatrixXd> weightsByLayer;
        int amountOfLayers = layerSizes.size();
        for (
            int layerPosition = 0;
            layerPosition < amountOfLayers - 1;
            ++layerPosition
        ) {
            int outgoingLayerSize = layerSizes[layerPosition];
            int incomingLayerSize = layerSizes[layerPosition + 1];

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
