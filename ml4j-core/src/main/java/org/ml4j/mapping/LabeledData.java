/*
 * Copyright 2014 the original author or authors.
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
package org.ml4j.mapping;

import java.io.Serializable;

/**
 * <p>Wrapper for data elements of type T with labels of type L.</p>
 *
 * @author Michael Lavelle
 */
public class LabeledData<T extends Serializable,L extends Serializable> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private T data;
	private L label;
	
	/**
	 * <p>Constructor for LabeledData.</p>
	 *
	 * @param data The data to be labeled
	 * @param label a label for the data
	 */
	public LabeledData(T data,L label)
	{
		this.data = data;
		this.label = label;
	}

	/**
	 * <p>Getter for the <code>data</code>.</p>
	 */
	public T getData() {
		return data;
	}

	/**
	 * <p>Getter for the <code>label</code>.</p>
	 */
	public L getLabel() {
		return label;
	}
	
	
	
}
