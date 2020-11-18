package edu.neural_network.layer;

import java.util.function.Function;

public interface Layer {

    int getNeuronCount();

    Function<Double, Double> getActivationFunction();

    double[] feedForward(double[] args);

    double[][] getWeightMatrix();

    double[] getThresholdVector();

    double[] getResult();
}
