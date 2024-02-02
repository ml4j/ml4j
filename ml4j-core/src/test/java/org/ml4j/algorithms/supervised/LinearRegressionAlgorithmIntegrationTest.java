/*
 * Copyright 2015 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.algorithms.supervised;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.algorithms.impl.RegularizationContext;
import org.ml4j.algorithms.impl.SimpleRegularizationContext;
import org.ml4j.algorithms.supervised.impl.LinearRegressionAlgorithmImpl;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.ml4j.util.NumericFeaturesMatrixCsvDataExtractor;


/**
 * Initial end-to-end integration test for numeric label prediction.
 *
 * Uses data from a programming assigment on Stanford's Machine Learning
 * Coursera course to take data consisting of attributes of houses and
 * applying Linear Regression to predict a price for the house.
 *
 * @author Michael Lavelle
 */
public class LinearRegressionAlgorithmIntegrationTest {

    private double[][] x;
    private double[] y;


    @BeforeEach
    public void setUp() throws Exception
    {
        // Read housing data used as part of a programming assignment on
        // Stanford's Machine Learning course
        DoubleArrayMatrixLoader matrixLoader = new DoubleArrayMatrixLoader(this.getClass().getClassLoader());
        double[][] data = matrixLoader.loadDoubleMatrixFromCsv("ex1data2.txt",new NumericFeaturesMatrixCsvDataExtractor(), 0, 47);
        // Split the data into feature matrix ( with intercept term of "1"), and a label vector
        x = new double[data.length][3];
        y = new double[data.length];
        for (int i = 0; i < data.length ;i++)
        {
            double[] vals = data[i];
            x[i] = new double[] {1,vals[0],vals[1]};
            y[i] = vals[2];
        }


        // Assert that we have read the training data correctly by checking first result and result count
        Assertions.assertEquals(47,x.length);
        Assertions.assertEquals(2104.0, x[0][1]);
        Assertions.assertEquals(1.0, x[0][0]);
        Assertions.assertEquals(3.0, x[0][2]);
        Assertions.assertEquals(399900.0, y[0]);


    }

    @Test
    public void testLabelPrediction_WithNormalEquationLinearRegressionAlgorithm()
    {

        // Create a linear regression algorithm that we can use to predict the price.
        LinearRegressionAlgorithm<RegularizationContext> linearRegressionAlgorithm =
                new LinearRegressionAlgorithmImpl();


        // Create a training context for our chosen algorithm, setting the necessary parameters
        RegularizationContext trainingContext = new SimpleRegularizationContext(0d);

        // Train the price predictor to learn from training set
        LinearRegressionHypothesisFunction hypothesisFunction = linearRegressionAlgorithm.getOptimalHypothesisFunction(x, y, trainingContext);


        // Predict a price for a specified house
        double[] houseFeatures = {1,1650,3};
        Number predictedPrice = hypothesisFunction.predict(houseFeatures);

        // Assert the price is not null
        Assertions.assertNotNull(predictedPrice, "Predicted price should not be null");

        // Assert the price matches the same price as predicted by the Octave
        // programming assignment from Stanford's Machine Learning course
        Assertions.assertEquals(293081, predictedPrice.intValue());
    }



}