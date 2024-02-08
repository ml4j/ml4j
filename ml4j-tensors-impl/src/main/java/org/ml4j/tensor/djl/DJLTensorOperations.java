package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import org.ml4j.tensor.Operatable;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorOperations;

public interface DJLTensorOperations extends TensorOperations<DJLTensorOperations>, Operatable<DJLTensorOperations, Size, DJLTensorOperations> {

	NDArray getNDArray();

	boolean isNativeGradient();
}
