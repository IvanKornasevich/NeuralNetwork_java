package edu;

import edu.function.Linear;
import edu.neural_network.NeuralNetwork;
import edu.neural_network.Perceptron;
import edu.neural_network.teacher.*;
import edu.neural_network.topology.PerceptronTopology;
import edu.neural_network.topology.Topology;

import java.util.*;
import java.util.stream.IntStream;

public class App {

    public static void main(String[] args) {

        var v0 = new double[]{0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1};
        var v1 = new double[]{1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1};
        var v2 = new double[]{0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1};
        var v3 = new double[]{1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1};

        Topology topology = new PerceptronTopology(v1.length);
        topology.addLayer(4, new Linear());

        NeuralNetwork nn = new Perceptron(topology);


        var cases = new ArrayList<LearnCase>(List.of(

                new PerceptronLearnCase(v0, new double[]{10, 0, 0, 0}),
                new PerceptronLearnCase(v1, new double[]{0, 10, 0, 0}),
                new PerceptronLearnCase(v2, new double[]{0, 0, 10, 0}),
                new PerceptronLearnCase(v3, new double[]{0, 0, 0, 10})

        ));

        var teacher = new PerceptronTeacher(nn, cases);

        teacher.learnCompetitive(1, 300000);

        System.out.println();

        var cCases0 = getCorruptedVector(cases.get(0), 10, 3);

        cCases0.forEach(x -> printArray(nn.run(x.args())));

    }

    static void printArray(double[] arr) {

        System.out.print("[ ");
        for (var i = 0; i < arr.length - 1; ++i) {
            System.out.printf("%.3f%s", arr[i], ", ");
        }
        System.out.printf("%.3f", arr[arr.length - 1]);
        System.out.print(" ]");
        System.out.println();
    }

    static List<LearnCase> getCorruptedVector(LearnCase learnCase, int vCount, int corruptionCount) {
        var rnd = new Random();
        var newSet = new ArrayList<LearnCase>();

        IntStream.range(0, vCount).forEach(i -> {

            var newV = new double[learnCase.args().length];
            System.arraycopy(learnCase.args(), 0, newV, 0, newV.length);

            IntStream.range(0, corruptionCount).forEach(j -> {

                var rInt = rnd.nextInt(newV.length);
                if (newV[rInt] > 0.5) {
                    newV[rInt] = 0;
                } else {
                    newV[rInt] = 1;
                }

            });
            newSet.add(new PerceptronLearnCase(newV, learnCase.results()));
        });
        return newSet;
    }
}
