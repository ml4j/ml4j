package org.ml4j.tensor.dl4j;

import org.ml4j.autograd.impl.GradNodeWrapper;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorWrapperImpl;
import org.ml4j.tensor.ml4j.ML4JFromDL4JTensorWrapperImpl;
import org.ml4j.tensor.ml4j.ML4JTensor;
import org.ml4j.tensor.ml4j.ML4JTensorOperations;
import org.ml4j.tensor.ml4j.ML4JTensorOperationsImpl;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JFromML4JTensorWrapperImpl extends TensorWrapperImpl<ML4JTensor, DL4JTensor, ML4JTensorOperations, DL4JTensorOperations> implements DL4JTensor {

    private DirectedComponentsContext directedComponentsContext;

    public DL4JFromML4JTensorWrapperImpl(ML4JTensor t) {
        super(t);
        this.directedComponentsContext = t.getDirectedComponentsContext();
    }

    @Override
    protected ML4JTensor extract(DL4JTensor tensor) {
        if(tensor instanceof DL4JFromML4JTensorWrapperImpl) {
            DL4JFromML4JTensorWrapperImpl oth = (DL4JFromML4JTensorWrapperImpl)tensor;
            return oth.t;
        }
        //return new DJLFromDL4JTensorWrapperImpl(tensor);
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    protected DL4JTensor create(ML4JTensor tensor) {
        /*
        if(tensor instanceof DJLFromDL4JTensorWrapperImpl) {
            DJLFromDL4JTensorWrapperImpl oth = (DJLFromDL4JTensorWrapperImpl)tensor;
            return oth.getT();
        }

         */
        return new DL4JFromML4JTensorWrapperImpl(tensor);
    }

    @Override
    protected DL4JTensorOperations createData(ML4JTensorOperations ml4JTensorOperations) {
        INDArray ndArray = Nd4j.create(ml4JTensorOperations.getDataAsFloatArray(),
                ml4JTensorOperations.size().dimensions());
        return new DL4JTensorOperationsImpl(ndArray);
    }

    @Override
    protected ML4JTensorOperations extractData(DL4JTensorOperations ml4JTensorOperations) {
        return new ML4JTensorOperationsImpl(directedComponentsContext, ml4JTensorOperations);
    }

    @Override
    public DL4JTensor get() {
        return this;
    }

    @Override
    public ValueNode<DL4JTensor> getValueNode() {
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new DL4JFromML4JTensorWrapperImpl(f).requires_grad_(false), f -> new ML4JFromDL4JTensorWrapperImpl(directedComponentsContext, f).requires_grad_(false));
    }

    @Override
    public GradNode<DL4JTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> f == null ? null : new DL4JFromML4JTensorWrapperImpl(f).requires_grad_(false), f -> f == null ? null : new ML4JFromDL4JTensorWrapperImpl(directedComponentsContext, f).requires_grad_(false));
    }

    @Override
    public DL4JTensor apply(DifferentiableUnaryOperator<DL4JTensor, DL4JTensorOperations, Size> differentiableUnaryOperator) {
        DL4JTensor tensor = new DL4JTensorImpl(this).apply(differentiableUnaryOperator);
        return tensor;
    }

    @Override
    public DL4JTensor apply(DifferentiableBinaryOperator<DL4JTensor, DL4JTensorOperations, Size> differentiableBinaryOperator, DL4JTensor ml4JTensor) {
        return new DL4JTensorImpl(this).apply(differentiableBinaryOperator, ml4JTensor);
    }

    @Override
    public INDArray getNDArray() {
        if (size().dimensions().length == 0) {
            return Nd4j.scalar(getDataAsFloatArray()[0]);
        } else {
            return Nd4j.create(getDataAsFloatArray(), size().dimensions());
        }
    }
}