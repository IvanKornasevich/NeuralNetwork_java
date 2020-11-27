package edu.neural_network;

import edu.neural_network.layer.Layer;
import edu.neural_network.topology.Topology;

import java.util.List;

public class Perceptron implements NeuralNetwork {

    private final List<Layer> layers;

    public Perceptron(Topology topology) {

        layers = topology.getLayers();
    }

    @Override
    public List<Layer> getLayers() {

        return layers;
    }

    @Override
    public double[] run(double[] args) {

        if (args.length != layers.get(0).getWeightsMatrix()[0].length) {
            throw new IllegalArgumentException("Arguments and first layer inputs count was not same");
        }

        for (var layer : layers) {
            args = layer.feedForward(args);
        }

        return args;
    }
}
