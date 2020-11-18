package edu;

import edu.neural_network.*;
import edu.neural_network.topology.*;

public class App {

    public static void main(String[] args) {

        Topology topology = new NetworkTopology(5);
        topology.addLayer(10, x -> x);
        topology.addLayer(10, x -> x);

        NeuralNetwork nn = new Perceptron(topology);

        var argsnn = new double[]{1, 2, 3, 4, 5};
        var result = nn.run(argsnn);

        printArray(argsnn);
        printArray(result);
    }

    static void printArray(double[] arr) {

        for (var i : arr) {
            System.out.println(String.format("%.5f", i));
        }
        System.out.println();
    }
}
