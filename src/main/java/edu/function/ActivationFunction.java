package edu.function;

public interface ActivationFunction {

    double activate(double arg);

    double derivativeThroughResult(double arg);
}
