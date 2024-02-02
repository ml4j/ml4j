package org.ml4j.tensor.djl;

import org.ml4j.tensor.Operatable;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorOperations;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

public interface DJLTensorOperations extends TensorOperations<DJLTensorOperations>, Operatable<DJLTensorOperations, Size, DJLTensorOperations> {

	NDArray getNDArray();

	boolean isNativeGradient();
}
