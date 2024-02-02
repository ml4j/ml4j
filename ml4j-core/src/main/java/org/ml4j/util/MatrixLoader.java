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

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Creates a collection of elements of type T from a file accessible via a class loader.
 * 
 * Uses a CsvDataExtractor to parse csv attributes into an element
 * 
 * @author Michael Lavelle
 */
public class MatrixLoader<T>  {

	private ClassLoader classLoader;
	
	/**
	 * @param filePath The file path relative to the classLoader from which to load the data
	 * @param classLoader A classloader to be used to load files
	 * @param dataExtractor A data extractor defining how to parse csv attributes into elements
	 */
	public MatrixLoader(ClassLoader classLoader) 
	{
		this.classLoader = classLoader;
	}
	
	/**
	 * Retrieve the elements from the csv file as a Collection
	 * 
	 * @param filePath The file path relative to the classLoader from which to load the data
	 * @param dataExtractor A data extractor defining how to parse csv attributes into elements
	 * @param startRowIndex The index of the first row to load
	 * @param endRowIndex The index acting as an upper bound on the row index
	 */
	public Collection<T> loadMatrixFromCsv(String filePath,CsvDataExtractor<T> dataExtractor,int startRowIndex,int endRowIndex)
	{
		Collection<T> data = new ArrayList<T>();
		InputStreamReader inputStreamReader = null;
		BufferedReader bufferedReader =null;
		InputStream is = classLoader.getResourceAsStream(filePath);
		try
		{
			if (is == null) throw new FileNotFoundException(filePath);
			inputStreamReader = new InputStreamReader(is);
			bufferedReader = new BufferedReader(inputStreamReader);
			String line = bufferedReader.readLine();
			int index = 0;
			while(line != null && index < endRowIndex)
			{
				if ( index >=startRowIndex)
				{
				if (line.trim().length() > 0)
				{
					String[] parts = line.split(",");
					T d = dataExtractor.createData(parts);
					data.add(d);
				}
				}
				line = bufferedReader.readLine();

				index++;
			}		
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
		finally
		{
			try {
				if (bufferedReader != null) bufferedReader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			try {
				if (inputStreamReader != null) inputStreamReader.close();
			} catch (IOException e) {}
			try {
				if (is != null) is.close();
			} catch (IOException e) {}
		}
		return data;
	}
	
}
