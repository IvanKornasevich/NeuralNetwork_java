package edu.neural_network.layer;

import edu.function.ActivationFunction;

public interface Layer {

    int neuronsCount();

    ActivationFunction activationFunction();

    double[] feedForward(double[] args);

    double[][] getWeightsMatrix();

    double[] getThresholdVector();

    double[] getResults();
}
