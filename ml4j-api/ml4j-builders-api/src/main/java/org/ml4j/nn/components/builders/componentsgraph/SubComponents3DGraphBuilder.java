/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.components.builders.componentsgraph;

import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;

public interface SubComponents3DGraphBuilder<C extends Axons3DBuilder<T>, D extends AxonsBuilder<T>, T extends NeuralComponent<?>>
		extends Components3DGraphBuilder<C, D, T> {

	PathEnder<C, SubComponents3DGraphBuilder<C, D, T>> endPath();

	SubComponents3DGraphBuilder<C, D, T> withPath();

}
