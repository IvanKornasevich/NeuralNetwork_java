package edu.neural_network.topology;

import edu.function.ActivationFunction;
import edu.neural_network.layer.Layer;
import edu.neural_network.layer.PerceptronLayer;

import java.util.LinkedList;
import java.util.List;

public class PerceptronTopology implements Topology {

    private final int argsCount;
    private final List<Layer> layers = new LinkedList<>();

    public PerceptronTopology(int argsCount) {

        this.argsCount = argsCount;
    }

    public void addLayer(int neuronesCount, ActivationFunction activationFunction) {

        Layer newLayer;

        if (layers.isEmpty()) {
            newLayer = new PerceptronLayer(argsCount, neuronesCount, activationFunction);
        } else {
            newLayer = new PerceptronLayer(layers.get(layers.size() - 1).neuronsCount(), neuronesCount, activationFunction);
        }

        layers.add(newLayer);
    }

    @Override
    public List<Layer> getLayers() {

        return layers;
    }
}
