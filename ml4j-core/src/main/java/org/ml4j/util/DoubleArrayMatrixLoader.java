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
package org.ml4j.util;

import java.util.Collection;
/**
 * Loads double[][] matrices from csv files on the classpath
 * 
 * @author Michael Lavelle
 *
 */
public class DoubleArrayMatrixLoader extends MatrixLoader<double[]> {

	public DoubleArrayMatrixLoader(ClassLoader classLoader) {
		super(classLoader);
	}
	
	public double[][] loadDoubleMatrixFromCsv(String filePath,CsvDataExtractor<double[]> dataExtractor,int startRowIndex,int endRowIndex)
	{
		Collection<double[]> results = super.loadMatrixFromCsv(filePath, dataExtractor,startRowIndex,endRowIndex);
		int rowCount = Math.min(endRowIndex - startRowIndex,results.size());
		return results.toArray(new double[rowCount][results.iterator().next().length] );
	}

}
