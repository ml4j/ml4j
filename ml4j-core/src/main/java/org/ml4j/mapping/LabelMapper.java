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
 * <p>Converts labels of type L to and from numeric vectors</p>
 *
 * @author Michael Lavelle
 */
public interface LabelMapper<L> extends Serializable {

	/**
	 * <p>toLabelVector.</p>
	 *
	 * @param label a label which is to be converted to a numeric vector
	 */
	public double[] toLabelVector(L label);
	/**
	 * <p>fromLabelVector.</p>
	 *
	 * @param labelVector a numeric vector to be converted to a label
	 */
	public L fromLabelVector(double[] labelVector);
	
	public int getVectorLength();
}
