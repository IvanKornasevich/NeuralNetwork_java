package edu.neural_network;

import edu.neural_network.layer.Layer;

import java.util.List;

public interface NeuralNetwork {

    List<Layer> getLayers();

    int argsCount();

    int resultsCount();

    double[] run(double[] args);
}
