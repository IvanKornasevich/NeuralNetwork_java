package edu.neural_network;

import edu.neural_network.layer.Layer;

import java.util.List;

public interface NeuralNetwork {

    List<Layer> getLayers();

    double[] run(double[] args);
}
