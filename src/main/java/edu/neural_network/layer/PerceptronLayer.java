package edu.neural_network.layer;

import edu.function.ActivationFunction;

import java.util.Random;

public class PerceptronLayer implements Layer {

    protected final double[] thresholdVector;

    protected final double[] resultVector;

    protected final ActivationFunction activationFunction;

    protected double[][] weightMatrix;

    public PerceptronLayer(int connectionsCount, int neuronsCount, ActivationFunction activationFunction) {

        weightMatrix = new double[neuronsCount][connectionsCount];
        var rnd = new Random();
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
    public ActivationFunction activationFunction() {

        return activationFunction;
    }

    @Override
    public double[] feedForward(double[] args) {

        for (int i = 0; i < weightMatrix.length; i++) {
            resultVector[i] = 0;

            for (int j = 0; j < weightMatrix[i].length; j++) {
                resultVector[i] += weightMatrix[i][j] * args[j];
            }

            resultVector[i] = activationFunction.activate(resultVector[i] - thresholdVector[i]);
        }

        return resultVector;
    }

    @Override
    public int neuronsCount() {

        return weightMatrix.length;
    }

    @Override
    public double[][] getWeightsMatrix() {

        return weightMatrix;
    }

    @Override
    public double[] getThresholdVector() {

        return thresholdVector;
    }

    @Override
    public double[] getResults() {

        return resultVector;
    }
}
