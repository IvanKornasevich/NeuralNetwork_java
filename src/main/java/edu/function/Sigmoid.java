package edu.function;


public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double arg) {
        return Math.exp(arg) / (Math.exp(arg) + 1);
    }

    @Override
    public double derivativeThroughResult(double arg) {
        return arg * (1 - arg);
    }
}
