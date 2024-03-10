#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using std::cout; using std::cin; using std::endl;
using std::string;
using std::vector;

class Network {
public:
    vector<int> layerSizes;

    Network(
        vector<int> ls
    ):
        layerSizes(ls)
    {
        vector<Eigen::MatrixXd> weightsByLayer;
        int amountOfLayers = layerSizes.size();
        for (
            int layerPosition = 0;
            layerPosition < amountOfLayers - 1;
            ++layerPosition
        ) {
            int outgoingLayerSize = layerSizes[layerPosition];
            int incomingLayerSize = layerSizes[layerPosition + 1];

            Eigen::MatrixXd layerWeights = Eigen::MatrixXd::Random(
                incomingLayerSize, outgoingLayerSize
            );
            Eigen::VectorXd layerBiases = Eigen::VectorXd::Random(
                incomingLayerSize
            );

            layerWeights = 0.5 * (
                layerWeights
                + Eigen::MatrixXd::Ones(
                    incomingLayerSize, outgoingLayerSize
                )
            );
            // TODO: Include optional division by square root of amount of outgoing weights
            layerBiases = 0.5 * (
                layerBiases + Eigen::VectorXd::Ones(incomingLayerSize)
            );

            weightsByLayer.push_back(layerWeights);
        }
    }
};
