package edu.neural_network.teacher;

import edu.neural_network.NeuralNetwork;
import edu.neural_network.layer.Layer;

import java.util.List;
import java.util.stream.IntStream;

public class PerceptronTeacher implements NeuralNetworkTeacher {

    private final NeuralNetwork neuralNetwork;
    private final List<LearnCase> learnSet;

    public PerceptronTeacher(NeuralNetwork neuralNetwork, List<LearnCase> learnSet) {

        this.neuralNetwork = neuralNetwork;
        this.learnSet = learnSet;
    }

    @Override
    public void Learn(double learnRate, int agesCount) {
        IntStream.range(0, agesCount)
                .forEach(i -> learnSet
                        .forEach(x -> {
                            if (x.args().length != neuralNetwork.argsCount()) {
                                throw new IllegalArgumentException("Learn case and neural network args count was not same");
                            }
                            if (x.results().length != neuralNetwork.resultsCount()) {
                                throw new IllegalArgumentException("Learn case and neural network results count was not same");
                            }
                            backProp(learnRate, x.args(), x.results(), neuralNetwork.run(x.args()));
                        }));
    }

    private void backProp(double learnRate, double[] args, double[] expectedResults, double[] actualResults) {

        var layers = neuralNetwork.getLayers();

        var curErrors = calculateLastLayerError(expectedResults, actualResults);

        for (var k = layers.size() - 1; k > 0; --k) {

            curErrors = backPropHiddenLayer(learnRate, layers.get(k - 1), layers.get(k), curErrors);
        }

        backPropFirstLayer(learnRate, args, layers.get(0), curErrors);

    }

    private double[] calculateLastLayerError(double[] expectedResults, double[] actualResults) {

        var lastLayer = neuralNetwork.getLayers().get(neuralNetwork.getLayers().size() - 1);
        var curErrors = new double[lastLayer.neuronsCount()];

        for (var i = 0; i < expectedResults.length; ++i) {
            curErrors[i] = actualResults[i] - expectedResults[i];
        }
        return curErrors;
    }

    private double[] backPropHiddenLayer(double learnRate, Layer prevLayer, Layer curLayer, double[] curLayerErrors) {

        var matrix = curLayer.getWeightsMatrix();
        var curLayerResults = curLayer.getResults();

        var prevLayerErrors = new double[prevLayer.neuronsCount()];

        for (var i = 0; i < matrix.length; ++i) {

            var weightDelta = curLayerErrors[i] *
                    curLayer.activationFunction().derivativeThroughResult(curLayerResults[i]);

            for (var j = 0; j < matrix[i].length; ++j) {

                matrix[i][j] -= learnRate * prevLayer.getResults()[j] * weightDelta;
                prevLayerErrors[j] += matrix[i][j] * weightDelta;

            }

            curLayer.getThresholdVector()[i] += learnRate * weightDelta;
        }

        return prevLayerErrors;
    }

    private void backPropFirstLayer(double learnRate, double[] args, Layer firstLayer, double[] firstLayerErrors) {

        var matrix = firstLayer.getWeightsMatrix();
        var firstLayerResults = firstLayer.getResults();

        for (var i = 0; i < matrix.length; ++i) {
            var weightDelta = firstLayerErrors[i] *
                    firstLayer.activationFunction().derivativeThroughResult(firstLayerResults[i]);
            for (var j = 0; j < matrix[i].length; ++j) {
                matrix[i][j] -= learnRate * args[j] * weightDelta;
            }
            firstLayer.getThresholdVector()[i] += learnRate * weightDelta;
        }
    }
}

