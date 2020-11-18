package edu.neural_network.layer;

import java.util.function.Function;

public class NetworkLayer implements Layer {

    protected final double[] thresholdVector;

    protected final double[] resultVector;

    protected final Function<Double, Double> activationFunction;

    protected double[][] weightMatrix;

    public NetworkLayer(int connectionsCount, int neuronsCount, Function<Double, Double> activationFunction) {

        weightMatrix = new double[neuronsCount][connectionsCount];
        for (int i = 0; i < weightMatrix.length; i++) {
            for (int j = 0; j < weightMatrix[0].length; j++) {
                weightMatrix[i][j] = 0.5;
            }
        }

        thresholdVector = new double[neuronsCount];
        resultVector = new double[neuronsCount];
        this.activationFunction = activationFunction;
    }

    @Override
    public Function<Double, Double> getActivationFunction() {

        return activationFunction;
    }

    @Override
    public double[] feedForward(double[] args) {

        for (int i = 0; i < weightMatrix.length; i++) {
            for (int j = 0; j < weightMatrix[0].length; j++) {
                resultVector[i] += weightMatrix[i][j] * args[j];
            }

            resultVector[i] -= thresholdVector[i];
            resultVector[i] = activationFunction.apply(resultVector[i]);
        }

        return resultVector;
    }

    @Override
    public int getNeuronCount() {

        return weightMatrix.length;
    }

    @Override
    public double[][] getWeightMatrix() {

        return weightMatrix;
    }

    @Override
    public double[] getThresholdVector() {

        return thresholdVector;
    }

    @Override
    public double[] getResult() {

        return resultVector;
    }
}
