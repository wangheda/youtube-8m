from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

# pylint: disable=protected-access
_state_size_with_prefix = rnn_cell_impl._state_size_with_prefix
# pylint: enable=protected-access


def _infer_state_dtype(explicit_dtype, state):
    """Infer the dtype of an RNN state.

    Args:
      explicit_dtype: explicitly declared dtype or None.
      state: RNN's hidden state. Must be a Tensor or a nested iterable containing
        Tensors.

    Returns:
      dtype: inferred dtype of hidden state.

    Raises:
      ValueError: if `state` has heterogeneous dtypes or is empty.
    """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to infer a "
                "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype

def _on_device(fn, device):
    """Build the subgraph defined by lambda `fn` on `device` if it's not None."""
    if device:
        with ops.device(device):
            return fn()
    else:
        return fn()

# pylint: disable=unused-argument
def _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, call_cell, state_size, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.

    Returns an (output, state) pair conditioned on the sequence_lengths.
    When skip_conditionals=False, the pseudocode is something like:

    if t >= max_sequence_length:
      return (zero_output, state)
    if t < min_sequence_length:
      return call_cell()

    # Selectively output zeros or output, old state or new state depending
    # on if we've finished calculating each row.
    new_output, new_state = call_cell()
    final_output = np.vstack([
      zero_output if time >= sequence_lengths[r] else new_output_r
      for r, new_output_r in enumerate(new_output)
    ])
    final_state = np.vstack([
      state[r] if time >= sequence_lengths[r] else new_state_r
      for r, new_state_r in enumerate(new_state)
    ])
    return (final_output, final_state)

    Args:
      time: Python int, the current time step
      sequence_length: int32 `Tensor` vector of size [batch_size]
      min_sequence_length: int32 `Tensor` scalar, min of sequence_length
      max_sequence_length: int32 `Tensor` scalar, max of sequence_length
      zero_output: `Tensor` vector of shape [output_size]
      state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
        or a list/tuple of such tensors.
      call_cell: lambda returning tuple of (new_output, new_state) where
        new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
        new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
      state_size: The `cell.state_size` associated with the state.
      skip_conditionals: Python bool, whether to skip using the conditional
        calculations.  This is useful for `dynamic_rnn`, where the input tensor
        matches `max_sequence_length`, and using conditionals just slows
        everything down.

    Returns:
      A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
        final_output is a `Tensor` matrix of shape [batch_size, output_size]
        final_state is either a single `Tensor` matrix, or a tuple of such
          matrices (matching length and shapes of input `state`).

    Raises:
      ValueError: If the cell returns a state tuple whose length does not match
        that returned by `state_size`.
    """

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        copy_cond = (time >= sequence_length)
        return _on_device(
            lambda: array_ops.where(copy_cond, output, new_output),
            device=new_output.op.device)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state

def dynamic_rnn(cell, inputs, inerval, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    This function is functionally identical to the function `rnn` above, but
    performs fully dynamic unrolling of `inputs`.

    Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`, one for
    each frame.  Instead, `inputs` may be a single `Tensor` where
    the maximum time is either the first or second dimension (see the parameter
    `time_major`).  Alternatively, it may be a (possibly nested) tuple of
    Tensors, each of them having matching batch and time dimensions.
    The corresponding output is either a single `Tensor` having the same number
    of time steps and batch size, or a (possibly nested) tuple of such tensors,
    matching the nested structure of `cell.output_size`.

    The parameter `sequence_length` is optional and is used to copy-through state
    and zero-out outputs when past a batch element's sequence length. So it's more
    for correctness than performance, unlike in rnn().

    Args:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.

        If `time_major == False` (default), this must be a `Tensor` of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such
          elements.

        If `time_major == True`, this must be a `Tensor` of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such
          elements.

        This may also be a (possibly nested) tuple of Tensors satisfying
        this property.  The first two dimensions must match across all the inputs,
        but otherwise the ranks and other shape components may differ.
        In this case, input to `cell` at each time-step will replicate the
        structure of these tuples, except for the time dimension (from which the
        time is taken).

        The input to `cell` at each time step will be a `Tensor` or (possibly
        nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
      sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
      A pair (outputs, state) where:

        outputs: The RNN output `Tensor`.

          If time_major == False (default), this will be a `Tensor` shaped:
            `[batch_size, max_time, cell.output_size]`.

          If time_major == True, this will be a `Tensor` shaped:
            `[max_time, batch_size, cell.output_size]`.

          Note, if `cell.output_size` is a (possibly nested) tuple of integers
          or `TensorShape` objects, then `outputs` will be a tuple having the
          same structure as `cell.output_size`, containing Tensors having shapes
          corresponding to the shape data in `cell.output_size`.

        state: The final state.  If `cell.state_size` is an int, this
          will be shaped `[batch_size, cell.state_size]`.  If it is a
          `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
          If it is a (possibly nested) tuple of ints or `TensorShape`, this will
          be a tuple having the corresponding shapes.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    # pylint: disable=protected-access
    if not isinstance(cell, rnn_cell_impl._RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    # pylint: enable=protected-access

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
        # (B,T,D) => (T,B,D)
        flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                           for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.get_shape())
        sequence_length = array_ops.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
        batch_size = input_shape[0][1]

        for input_ in input_shape:
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError("All inputs should have the same batch size")

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            inerval,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            flat_output = nest.flatten(outputs)
            flat_output = [array_ops.transpose(output, [1, 0, 2])
                           for output in flat_output]
            outputs = nest.pack_sequence_as(
                structure=outputs, flat_sequence=flat_output)

        return (outputs, final_state)

def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      interval,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
    """Internal implementation of Dynamic RNN.

    Args:
      cell: An instance of RNNCell.
      inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
        tuple of such elements.
      initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
        `cell.state_size` is a tuple, then this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      parallel_iterations: Positive Python int.
      swap_memory: A Python boolean
      sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
      dtype: (optional) Expected dtype of output. If not specified, inferred from
        initial_state.

    Returns:
      Tuple `(final_outputs, final_state)`.
      final_outputs:
        A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
        `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
        objects, then this returns a (possibly nsted) tuple of Tensors matching
        the corresponding shapes.
      final_state:
        A `Tensor`, or possibly nested tuple of Tensors, matching in length
        and shapes to `initial_state`.

    Raises:
      ValueError: If the input depth cannot be inferred via shape inference
        from the inputs.
    """
    pre_state = initial_state
    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _state_size_with_prefix(size, prefix=[batch_size])
        return array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i,
                                 _infer_state_dtype(dtype, state))
                      for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state, pre_state):
        """Take a time step of the dynamic RNN.

        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.

        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        new_pre_state = []
        input_state = []
        for cstate, pstate in zip(list(pre_state),list(state)):
            c1 = tf.cond(tf.logical_and(tf.equal(tf.mod(time,interval),tf.constant(0)),
                        tf.greater(time, tf.constant(0))), lambda: (cstate.c+pstate.c)/2, lambda: pstate.c)
            h1 = tf.cond(tf.logical_and(tf.equal(tf.mod(time,interval),tf.constant(0)),
                        tf.greater(time, tf.constant(0))), lambda: (cstate.h+pstate.h)/2, lambda: pstate.h)
            c2 = tf.cond(tf.logical_and(tf.equal(tf.mod(time,interval),tf.constant(0)),
                        tf.greater(time, tf.constant(0))), lambda: pstate.c, lambda: cstate.c)
            h2 = tf.cond(tf.logical_and(tf.equal(tf.mod(time,interval),tf.constant(0)),
                        tf.greater(time, tf.constant(0))), lambda: pstate.h, lambda: cstate.h)
            new_pre_state.append(tf.contrib.rnn.LSTMStateTuple(c2,h2))
            input_state.append(tf.contrib.rnn.LSTMStateTuple(c1,h1))

        new_pre_state = tuple(new_pre_state)
        input_state = tuple(input_state)

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, input_state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=input_state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        # Pack state if using state tuples
        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))

        return (time + 1, output_ta_t, new_state, new_pre_state)

    _, output_final_ta, final_state, final_pre_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state, pre_state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _state_size_with_prefix(
            output_size, prefix=[const_time_steps, const_batch_size])
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state)