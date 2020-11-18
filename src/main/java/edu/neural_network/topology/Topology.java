package edu.neural_network.topology;

import edu.neural_network.layer.Layer;

import java.util.List;
import java.util.function.Function;

public interface Topology {

    void addLayer(int neuronesCount, Function<Double, Double> activationFunction);

    List<Layer> getLayers();
}
