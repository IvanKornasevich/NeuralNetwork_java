package edu.neural_network.teacher;

public class PerceptronLearnCase implements LearnCase {

    private final double[] args;
    private final double[] results;

    public PerceptronLearnCase(double[] args, double[] results) {

        this.args = args;
        this.results = results;
    }


    @Override
    public double[] args() {
        return args;
    }

    @Override
    public double[] results() {
        return results;
    }
}
