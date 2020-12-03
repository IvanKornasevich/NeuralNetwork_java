package edu.neural_network.teacher;

public interface NeuralNetworkTeacher {

    void learnBackProp(double learnRate, int agesCount);

    void learnCompetitive(double learnRate, int agesCount);
}
