package edu;

import edu.function.Sigmoid;
import edu.neural_network.NeuralNetwork;
import edu.neural_network.Perceptron;
import edu.neural_network.teacher.PerceptronLearnCase;
import edu.neural_network.teacher.PerceptronTeacher;
import edu.neural_network.topology.PerceptronTopology;
import edu.neural_network.topology.Topology;

import java.util.List;

public class App {

    public static void main(String[] args) {


        Topology topology = new PerceptronTopology(2, 2);
        topology.addLayer(2, new Sigmoid());
        topology.addLayer(2, new Sigmoid());
        topology.addLayer(2, new Sigmoid());


        NeuralNetwork nn = new Perceptron(topology);

        var lc = new PerceptronLearnCase(new double[]{1, 2}, new double[]{3, 4});

        var teacher = new PerceptronTeacher(nn, List.of(lc));

        for (var i = 0; i < 10; ++i) {
            teacher.Learn(0.1, 50);
            printArray(nn.run(lc.args()));
        }

    }

    static void printArray(double[] arr) {

        for (var i : arr) {
            System.out.printf("%.5f%n", i);
        }
        System.out.println();
    }
}
