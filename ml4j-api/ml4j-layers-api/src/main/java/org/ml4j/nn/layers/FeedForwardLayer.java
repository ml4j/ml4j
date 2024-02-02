/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.layers;

import org.ml4j.nn.axons.Axons;

/**
 * A FeedForwardLayer is a DirectedLayer which composes input neurons and output neurons into a
 * directed acyclic bipartite graph.
 * There are no input-input connections or output-output connections, only input-output connections.
 * 
 * @author Michael Lavelle
 *
 * @param <A> The type of Axons in this FeedForwardLayer
 * @param <L> The type of FeedForwardLayer
 */
public interface FeedForwardLayer<A extends Axons<?, ?, ?>, L extends FeedForwardLayer<A, L>> 
    extends DirectedLayer<A, L> {
}
