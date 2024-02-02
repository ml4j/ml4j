package org.ml4j.tensor;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.arithmetic.operations.DifferentiableWrappedArithmeticOperations;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.impl.AutogradValueProperties;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;

public abstract class DifferentiableWrappedTensorOperations<V extends Tensor<V, D> & Value<V, D, Size>, D extends TensorOperations<D>> extends AutogradValueImpl<V, D, Size> implements AutogradValue<V, D, Size>, TensorOperations<V>, org.ml4j.autograd.DataSupplier<D>, Tensor<V, D>, DifferentiableWrappedArithmeticOperations<V, D, Size> {

    public DifferentiableWrappedTensorOperations(Supplier<D> data, AutogradValueProperties<Size> properties) {
        super(properties, data);
    }

    public <X extends AutogradValue<X, Y, Z>, Y, Z> DifferentiableWrappedTensorOperations(AutogradValue<X, Y, Z> other, Function<Y, D> dataMapper, Function<Z, Size> contextMapper, Function<X, V> valueMapper, Function<V, X> valueReverseMapper, Supplier<Optional<V>> nativeGradientSupplier) {
        super(other, dataMapper, contextMapper, valueMapper, valueReverseMapper, nativeGradientSupplier);
    }

    public DifferentiableWrappedTensorOperations(V other) {
        super(other);
    }

    @Override
    public V add(V other) {
        return applyBinaryOperator(other, D::add, (g, p) -> g, (g, p) -> g, "add:" + size() + ":" + other.context(), (f, s) -> getMappedContext(f, s));
    }

    @Override
    public V mul(V other) {
        return applyBinaryOperator(other, D::mul, (g, p) -> g.mul(p.getRight()), (g, p) -> g.mul(p.getLeft()), "mul:" + size() + ":" + other.context(), (f, s) -> getMappedContext(f, s));
    }

    @Override
    public boolean isNativeGradient() {
        return data().get().isNativeGradient();
    }

    @Override
    public V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis, BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<Size> contextMapper) {
        if (!size().getDimensions().equals(other.size().getDimensions())) {
            try {
                Size broadcastSize = MultiplicationRules.getBroadcast(size(), other.size());
                if (broadcastSize.getDimensions().equals(size().getDimensions()) || broadcastSize.getDimensions().equals(other.size().getDimensions())) {
                    float scale1 = (float) broadcastSize.numel() / (float) size().numel();
                    float scale2 = (float) broadcastSize.numel() / (float) other.size().numel();
                    return super.applyBinaryOperator(other, forward, (g, p) -> getSub(backThis.apply(g, p), size(), scale1).mul(scale1), (g, p) -> getSub(backOther.apply(g, p), other.size(), scale2).mul(scale2), op, (f, s) -> broadcastSize);
                } else {
                    throw new IllegalStateException();
                }
            } catch (IllegalArgumentException e) {
                // Size cannot be broadcast, perhaps it's matMul.
            }
        }
        return super.applyBinaryOperator(other, forward, backThis, backOther, op, contextMapper);
    }

    protected abstract V getSub(V other, Size size, float scale);

    @Override
    public V relu() {
        return applyUnaryOperator(D::relu, (g, v) -> g.mul(v.gt(0)), "gt", s -> s);
    }

    @Override
    public V view(Size size) {
        if (size.numel() != numel()) {
            throw new IllegalArgumentException("Number of elements do not match");
        }
        return applyUnaryOperator(v -> v.view(size), (g, v) -> g.view(size()), "view", s -> size);
    }

    @Override
    public V norm() {
        return applyUnaryOperator(t -> t.norm(), (g, v) -> backwardNotYetImplemented("norm"), "norm", s -> new Size());
    }

    @Override
    public V sum(int...axes) {
        return applyUnaryOperator(t -> t.sum(axes), (g, v) -> { if (axes.length > 0) throw new UnsupportedOperationException(); return v.mul(0).add(1).mul(g); }, "sum", s -> (axes.length == 0) ? new Size() : (axes.length == 1 && axes[0] == 0) ? new Size(1, size().dimensions()[1]) : new Size(size().dimensions()[0], 1) );
    }

    @Override
    public V argMax(int i) {
        return applyUnaryOperator(t -> t.argMax(i), (g, v) -> backwardNotYetImplemented("argMax"), "sum", s -> new Size() );
    }

    @Override
    public V argMax() {
        return applyUnaryOperator(t -> t.argMax(), (g, v) -> backwardNotYetImplemented("argMax"), "sum", s -> new Size() );
    }

    @Override
    public V mean() {
        return applyUnaryOperator(t -> t.mean(), (g, v) -> { return v.mul(0).add(1).mul(g).div(v.numel()); }, "mean", s -> new Size());
    }

    public V backwardNotYetImplemented(String op) {
        throw new UnsupportedOperationException(op);
    }

    public V t() {
        return applyUnaryOperator(D::t, (g, v) -> g.t(), "t", s -> s.t());
    }

    @Override
    public V sigmoid() {
        return applyUnaryOperator(D::sigmoid, (g, v) -> g.mul(sigGrad(v.getDataAsFloatArray()[0])), "sigmoid", s -> s);
    }

    @Override
    public V exp() {
        return applyUnaryOperator(D::exp, (g, v) -> backwardNotYetImplemented("exp"), "exp", s -> s);
    }

    @Override
    public V log() {
        return applyUnaryOperator(D::log, (g, v) -> backwardNotYetImplemented("log"), "log", s -> s);
    }


    @Override
    public V getTensor(int...indexes) {
        int[] inds = new int[indexes.length];
        int nonOneDimensions = 0;
        for (int i = 0; i < indexes.length; i++) {
            inds[i] = indexes[i] == -1 ? size().dimensions()[i] : 1;
            if (inds[i] != 1) {
                nonOneDimensions ++;
            }
        }
        int[] indsNonOne = new int[nonOneDimensions];
        int ind = 0;
        for (int i = 0; i < indexes.length; i++) {
            if (inds[i] != 1) {
                indsNonOne[ind] = inds[i];
                ind++;
            }
        }

        return applyUnaryOperator(t -> t.getTensor(indexes), (g, v) -> backwardNotYetImplemented("getTensor"), "getTensor", s -> new Size(indsNonOne));
    }



    @Override
    public V getTensor(int[]...ranges) {

        Pair<int[], int[][]> indsAll = getTensorRanges(size(), ranges);
        int[] inds = indsAll.getLeft();
        int[][] inds2 = indsAll.getRight();
        return applyUnaryOperator(t -> t.getTensor(ranges), (g, v) -> {V a = createAutogradValue(additiveIdentity(), new AutogradValueProperties<Size>().setContext(v.context()).setRegistry(properties().getRegistry()).setChildren(g.getGradNode().prev()).setNext(g.getGradNode().next()).setRequires_grad(g.requires_grad()).setCreate_graph(g.create_graph()).setName("rangeBackward")); var b = g.getTensor(ranges);  a.putTensor(b, ranges); return a; }, "getTensor", s -> new Size(inds));
    }

    private Pair<int[], int[][]> getTensorRanges(Size size, int[]...ranges) {

        int[] inds = new int[ranges.length];
        int[][] inds2 = new int[inds.length][2];
        List<Integer> is = new ArrayList<>();
        for (int i = 0; i < ranges.length; i++) {
            if (ranges[i][0] == -1 && ranges[i][1] == -1) {
                inds[i] = size.dimensions()[i];
                is.add(inds[i]);
                inds2[i][0] = 0;
                inds2[i][1] = size.dimensions()[i];;
            } else {
                int l = ranges[i][0] == -1 ? 0 : (ranges[i][0]);
                int r = ranges[i][1] == -1 ? size.dimensions()[i] : (ranges[i][1]);
                inds2[i][0] = l;
                inds2[i][1] = r;
                inds[i] = r - l;
                is.add(inds[i]);
            }

        }
        return new ImmutablePair<>(inds, inds2);
    }



    @Override
    public void putTensor(V tensor, int[]...ranges) {

        int[] inds = new int[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            if (ranges[i][0] == -1 && ranges[i][1] == -1) {
                inds[i] = size().dimensions()[i];
            } else {
                int l = ranges[i][0] == -1 ? 0 : (ranges[i][0]);
                int r = ranges[i][1] == -1 ? size().dimensions()[i] : (ranges[i][1]);
                inds[i] = r - l;
            }

        }
        applyInlineUnaryOperator(t -> { t.putTensor(tensor.data().get(), ranges); return t; }, "putTensor");
    }

    @Override
    public void putTensor(V tensor, int...indexes) {
        applyInlineUnaryOperator(t -> { t.putTensor(tensor.data().get(), indexes); return t; }, "putTensor");
    }

    @Override
    public V cloneTensor() {
        return applyUnaryOperator(D::cloneTensor, (g, v) -> this.backwardNotYetImplemented("cloneTensor"), "cloneTensor", s -> s);
    }

    @Override
    public V reshape(Size size) {
        V result = applyUnaryOperator(f -> f.reshape(size), (g, v) -> g.reshape(size()), "reshape", s -> size);
        result.properties().addLink(this.getValueNode());
        this.properties().addLink(result.getValueNode());
        return result;
    }

    @Override
    public V resize_(Size size) {
        V result = applyInlineUnaryOperator(t -> t.resize_(size), "resize_");
        this.properties().setContext(size);
        return result;
    }

    @Override
    public V mul_(V other) {
        return applyInlineBinaryOperator(other, D::mul_, "mul_");
    }

    @Override
    public V div_(V other) {
        return applyInlineBinaryOperator(other, D::div_, "div_");
    }

    @Override
    public V mul_(float v) {
        return applyInlineUnaryOperator(t -> t.mul_(v), "mul");
    }

    @Override
    public V div_(float v) {
        return applyInlineUnaryOperator(t -> t.div_(v), "div");
    }

    @Override
    public V sub_(float v) {
        return applyInlineUnaryOperator(t -> t.sub_(v), "sub");
    }

    @Override
    public V add_(float v) {
        return applyInlineUnaryOperator(t -> t.add_(v), "add");
    }

    @Override
    public void put(int index, float v) {
        applyInlineUnaryOperator(t -> { t.put(index, v); return t; }, "put");
    }

    @Override
    public void put(float v, int...indexes) {
        applyInlineUnaryOperator(t -> { t.put(v, indexes); return t; }, "put");
    }

    @Override
    public float get(int index) {
        return data().get().get(index);
    }

    @Override
    public float get(int... indexes) {
        return data().get().get(indexes);
    }

    @Override
    public V matmul(V other) {

        Size[] sizes = MultiplicationRules.matmul(size(), other.size());

        return this.applyBinaryOperator(other, (f, s) -> f.reshape(sizes[0]).matmul(s.reshape(sizes[1])).reshape(sizes[3]), (g, p) -> {
            return g.reshape(sizes[2]).matmul(p.getRight().reshape(sizes[1]).t()).reshape(size());
        }, (g, p) -> {
            return g.reshape(sizes[2]).t().matmul(p.getLeft().reshape(sizes[0])).t().reshape(other.size());
        }, "matmul", (f, s) -> {
            Size result =  sizes[3];
            int[] dims = result.dimensions();
            int [] firstDims = new int[dims.length- 1];
            for (int i = 0; i < firstDims.length; i++) {
                firstDims[i] = dims[i];
            }
            return new Size(new Size(firstDims), new Size(dims[dims.length - 1]));
        });
    }

    @Override
    public Size getMappedContext(Size f, Size s) {
        return MultiplicationRules.getBroadcast(f, s);
    }

    @Override
    public V bernoulli() {
        return applyUnaryOperator(D::bernoulli, (g, v) -> g, "gt", s -> s);
    }

    @Override
    public int numel() {
        return size().numel();
    }

    @Override
    public V columnSums() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V normal_(float v1, float v2) {

        return applyInlineUnaryOperator(t -> t.normal_(v1, v2), "normal");
    }

    @Override
    public V fill_(float value) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int size(int dim) {
        if (dim == -1) {
            return size().numel();
        } else {
            return size().getDimensions().get(dim);
        }
    }
    @Override
    public Size size() {
        return context();
    }

    @Override
    public V zero_() {
        return applyInlineUnaryOperator(t -> t.zero_(), "zero");
    }

    @Override
    public V view(int... dims) {
        if (dims.length == 1 && dims[0] == -1) {
            return applyUnaryOperator(t -> t.view(-1), (g, v) -> g.view(size()), "view", s -> new Size(s.numel()));
        }
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V rowSums() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public float sig(float x) {
        return 1f / (1f + (float)Math.exp(-x));
    }

    public float sigGrad(float x) {
        float s = sig(x);
        return s * ( 1 - s);
    }

    @Override
    public float[] getDataAsFloatArray() {
        return data().get().getDataAsFloatArray();
    }
}
