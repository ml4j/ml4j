/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.quickstart.sessions.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.layers.DefaultDirectedLayerFactory;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactoryImpl;
import org.ml4j.nn.supervised.DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.DefaultSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

/**
 * Quick-start implementation of DefaultSessionFactory, pre-configured with
 * default MatrixFactory, AxonsFactory, DifferentiableActivationFunctionFactory
 * and DirectedComponentFactory implementations.
 * 
 * @author Michael Lavelle
 *
 */
public class QuickstartSessionFactory extends DefaultSessionFactoryImpl {

	public static final MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();

	public static final AxonsFactory DEFAULT_AXONS_FACTORY = createDefaultAxonsFactory(DEFAULT_MATRIX_FACTORY);

	public static final DifferentiableActivationFunctionFactory DEFAULT_ACTIVATION_FUNCTION_FACTORY = new DefaultDifferentiableActivationFunctionFactory();	
	
	/**
	 * Quickstart session factory with all factories set to defaults
	 */
	public QuickstartSessionFactory(boolean isTrainingContext) {
		this(DEFAULT_MATRIX_FACTORY, createDefaultDirectedComponentsContext(DEFAULT_MATRIX_FACTORY, isTrainingContext));
	}
	
	/**
	 * Quickstart session factory with all factories set to defaults except for matrix factory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, boolean isTrainingContext) {
		this(matrixFactory, createDefaultDirectedComponentsContext(matrixFactory, isTrainingContext));
	}
	
	/**
	 * Quickstart session factory with all factories set to defaults, but with provided directedComponentsContext
	 */
	public QuickstartSessionFactory(DirectedComponentsContext directedComponentsContext) {
		this(directedComponentsContext.getMatrixFactory(), directedComponentsContext);
	}
	
	/**
	 * Quickstart session factory with specified MatrixFactory and DirectedComponentsContext, with all other
	 * factories set to defaults.
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, DirectedComponentsContext directedComponentsContext) {
		this(matrixFactory, createDefaultAxonsFactory(matrixFactory), DEFAULT_ACTIVATION_FUNCTION_FACTORY, directedComponentsContext);
	}

	/**
	 * Quickstart session factory with specified MatrixFactory and
	 * DifferentiableActivationFunctionFactory, with all other factories set to
	 * defaults.
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory,  DirectedComponentsContext directedComponentsContext) {
		this(matrixFactory, createDefaultAxonsFactory(matrixFactory), activationFunctionFactory, directedComponentsContext);
	}

	/**
	 * Quickstart session factory with specified MatrixFactory and AxonsFactory,
	 * with all other factories set to defaults.
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, AxonsFactory axonsFactory, DirectedComponentsContext directedComponentsContext) {
		this(matrixFactory, axonsFactory,
				createDefaultDirectedComponentFactory(matrixFactory, axonsFactory, DEFAULT_ACTIVATION_FUNCTION_FACTORY, directedComponentsContext),
				DEFAULT_ACTIVATION_FUNCTION_FACTORY, directedComponentsContext);
	}

	/**
	 * Quickstart session factory with specified MatrixFactory, AxonsFactory and
	 * DifferentiableActivationFunctionFactory, with all other factories set to
	 * defaults.
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory, DirectedComponentsContext directedComponentsContext) {
		this(matrixFactory, axonsFactory,
				createDefaultDirectedComponentFactory(matrixFactory, axonsFactory, activationFunctionFactory, directedComponentsContext),
				activationFunctionFactory, directedComponentsContext);
	}

	/**
	 * Quickstart session factory with specified MatrixFactory, AxonsFactory,
	 * DifferentiableActivationFunctionFactory and DirectedComponentFactory
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, AxonsFactory axonsFactory,
			DirectedComponentFactory directedComponentFactory, 
			DifferentiableActivationFunctionFactory activationFunctionFactory, DirectedComponentsContext directedComponentsContext) {
		super(matrixFactory, directedComponentFactory,
				createDefaultDirectedLayerFactory(axonsFactory, activationFunctionFactory, directedComponentFactory, 
						directedComponentsContext),
				new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory),
				new DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory), directedComponentsContext);
	}

	/**
	 * Quickstart session factory allowing all factories to be specified
	 * 
	 * @param matrixFactory
	 */
	public QuickstartSessionFactory(MatrixFactory matrixFactory, DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			LayeredSupervisedFeedForwardNeuralNetworkFactory layeredSupervisedFeedForwardNeuralNetworkFactory, 
			DirectedComponentsContext directedComponentsContext) {
		super(matrixFactory, directedComponentFactory, directedLayerFactory, supervisedFeedForwardNeuralNetworkFactory,
				layeredSupervisedFeedForwardNeuralNetworkFactory, directedComponentsContext);
	}

	private static DirectedComponentsContext createDefaultDirectedComponentsContext(MatrixFactory matrixFactory, boolean isTrainingContext) {
		return new DirectedComponentsContextImpl(matrixFactory, isTrainingContext);
	}

	private static AxonsFactory createDefaultAxonsFactory(MatrixFactory matrixFactory) {
		return new DefaultAxonsFactoryImpl(matrixFactory);
	}

	private static DirectedComponentFactory createDefaultDirectedComponentFactory(MatrixFactory matrixFactory,
			AxonsFactory axonsFactory, DifferentiableActivationFunctionFactory activationFunctionFactory, DirectedComponentsContext directedComponentsContext) {
		return new DefaultDirectedComponentFactoryImpl(matrixFactory, axonsFactory, activationFunctionFactory, directedComponentsContext);
	}

	private static DirectedLayerFactory createDefaultDirectedLayerFactory(AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory,
			DirectedComponentFactory directedComponentFactory, DirectedComponentsContext directedComponentsContext) {
		return new DefaultDirectedLayerFactory(axonsFactory, activationFunctionFactory, directedComponentFactory, directedComponentsContext);
	}

}
