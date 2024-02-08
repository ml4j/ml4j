package org.ml4j.tensor.ml4j;

import org.ml4j.autograd.impl.GradNodeWrapper;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorWrapperImpl;
import org.ml4j.tensor.dl4j.DL4JFromML4JTensorWrapperImpl;
import org.ml4j.tensor.dl4j.DL4JTensor;
import org.ml4j.tensor.dl4j.DL4JTensorOperations;
import org.ml4j.tensor.dl4j.DL4JTensorOperationsImpl;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ML4JFromDL4JTensorWrapperImpl extends TensorWrapperImpl<DL4JTensor, ML4JTensor, DL4JTensorOperations, ML4JTensorOperations> implements ML4JTensor {

    private DirectedComponentsContext context;

    public ML4JFromDL4JTensorWrapperImpl(DirectedComponentsContext context, DL4JTensor t) {
        super(t);
        this.context = context;
    }

    @Override
    public DirectedComponentsContext getDirectedComponentsContext() {
        return context;
    }

    @Override
    protected DL4JTensor extract(ML4JTensor tensor) {
        if(tensor instanceof ML4JFromDL4JTensorWrapperImpl) {
            ML4JFromDL4JTensorWrapperImpl oth = (ML4JFromDL4JTensorWrapperImpl)tensor;
            return oth.t;
        }
        return new DL4JFromML4JTensorWrapperImpl(tensor);
    }



    @Override
    protected ML4JTensor create(DL4JTensor tensor) {
        if(tensor instanceof DL4JFromML4JTensorWrapperImpl) {
            DL4JFromML4JTensorWrapperImpl oth = (DL4JFromML4JTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new ML4JFromDL4JTensorWrapperImpl(context, tensor);
    }

    @Override
    protected ML4JTensorOperations createData(DL4JTensorOperations data) {
        return new ML4JTensorOperationsImpl(context, data);
    }


    @Override
    protected DL4JTensorOperations extractData(ML4JTensorOperations ml4JTensorOperations) {
        INDArray ndArray = Nd4j.create(ml4JTensorOperations.getDataAsFloatArray(),
                ml4JTensorOperations.size().dimensions());
        return new DL4JTensorOperationsImpl(ndArray);
    }




    @Override
    public ML4JTensor get() {
        return this;
    }

    @Override
    public ValueNode<ML4JTensor> getValueNode() {
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new ML4JFromDL4JTensorWrapperImpl(context, f).requires_grad_(false), f -> new DL4JFromML4JTensorWrapperImpl(f).requires_grad_(false));
    }

    @Override
    public GradNode<ML4JTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> f == null ? null : new ML4JFromDL4JTensorWrapperImpl(context, f).requires_grad_(false), f -> f == null ? null : new DL4JFromML4JTensorWrapperImpl(f).requires_grad_(false));
    }

    @Override
    public ML4JTensor apply(DifferentiableUnaryOperator<ML4JTensor, ML4JTensorOperations, Size> differentiableUnaryOperator) {
        ML4JTensor tensor = new ML4JTensorImpl(this, context).apply(differentiableUnaryOperator);
        return tensor;
    }

    @Override
    public ML4JTensor apply(DifferentiableBinaryOperator<ML4JTensor, ML4JTensorOperations, Size> differentiableBinaryOperator, ML4JTensor ml4JTensor) {
        return new ML4JTensorImpl(this, context).apply(differentiableBinaryOperator, ml4JTensor);
    }


}