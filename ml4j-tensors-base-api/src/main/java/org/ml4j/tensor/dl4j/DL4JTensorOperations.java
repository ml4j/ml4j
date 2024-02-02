package org.ml4j.tensor.dl4j;

import org.ml4j.tensor.Operatable;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorOperations;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface DL4JTensorOperations extends TensorOperations<DL4JTensorOperations>, Operatable<DL4JTensorOperations, Size, DL4JTensorOperations> {

	INDArray getNDArray();
}
