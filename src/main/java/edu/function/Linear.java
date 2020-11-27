package edu.function;

public class Linear implements ActivationFunction {
    @Override
    public double activate(double arg) {
        return arg;
    }

    @Override
    public double derivativeThroughResult(double arg) {
        return 1;
    }
}
