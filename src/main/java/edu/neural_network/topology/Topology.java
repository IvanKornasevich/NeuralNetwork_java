package edu.neural_network.topology;

import edu.function.ActivationFunction;
import edu.neural_network.layer.Layer;

import java.util.List;

public interface Topology {

    void addLayer(int neuronesCount, ActivationFunction activationFunction);

    List<Layer> getLayers();
}
