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
package org.ml4j.algorithms.impl;

/**
 * Defines ConvergenceCriteria for an algorithm using runtime information encapsulated within training context of type C
 * 
 * @author Michael Lavelle
 */
public interface ConvergenceCriteria<C> {

	boolean isPrerequisiteConditionViolated(C trainingContext);

	boolean isConvergenceCompleteConditionSatisfied(C trainingContext);
	
}
