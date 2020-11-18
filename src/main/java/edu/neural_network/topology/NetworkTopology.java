package edu.neural_network.topology;

import edu.neural_network.layer.*;

import java.util.*;
import java.util.function.Function;

public class NetworkTopology implements Topology {

    private final int argsCount;
    private final List<Layer> layers = new ArrayList<>();

    public NetworkTopology(int argsCount) {

        this.argsCount = argsCount;
    }

    public void addLayer(int neuronesCount, Function<Double, Double> activationFunction) {

        Layer newLayer;

        if (layers.isEmpty()) {
            newLayer = new NetworkLayer(argsCount, neuronesCount, activationFunction);
        } else {
            newLayer = new NetworkLayer(layers.get(layers.size() - 1).getNeuronCount(), neuronesCount, activationFunction);
        }

        layers.add(newLayer);
    }

    @Override
    public List<Layer> getLayers() {

        return layers;
    }
}
