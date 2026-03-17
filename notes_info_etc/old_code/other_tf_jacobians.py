import tensorflow as tf

class JacobianWrapper(tf.Module):
    def jacobian_split(self, x):
        """Jacobian subgraph"""
        x.set_shape([1, 36])
        with tf.GradientTape(persistent=True) as tape: #, watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.model(x, training=False)
            y_unstacked = tf.unstack(y, axis=1)
        
        def temp_grad(y_ind):
            y_slice = y_unstacked[y_ind]
            return tape.gradient(y_slice, x)

        jacobian = tf.stack([
            temp_grad(0), temp_grad(1), temp_grad(2), temp_grad(3),
            temp_grad(4), temp_grad(5)
        ], axis=1)
        return jacobian

    def jacobian_forward(self, x):
        x.set_shape([1, 36])
        jacobian_cols = []
        for i in range(36):
            tangent = tf.one_hot(indices=i, depth=36, on_value=1, off_value=0, dtype=tf.float32)
            tangent = tf.reshape(tangent, (1, 36))
            
            with tf.autodiff.ForwardAccumulator(x, tangent) as acc:
                y = self.model(x, training=False)
                
            col = acc.jvp(y)
            jacobian_cols.append(col)
            
        jacobian = tf.stack(jacobian_cols, axis=2)
        
        return jacobian

    def jacobian_split2(self, x):
        x.set_shape([1, 36])
        
        projected_scalars = []
        grads = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            # Run inference
            y = self.model(x, training=False)
            
            # y needs to be static for the loop
            # Assuming y is (1, OutputDim)
            output_dim = 6 # or y.shape[1] if known

            for i in range(output_dim):
                # --- THE FIX ---
                # Create the "Selector" as a constant vector, not a slice operation.
                # Shape: (36, 1)
                # We create this outside the tape so it's just a constant in the graph.
                projection_vec = tf.reshape(
                    tf.one_hot(i, depth=output_dim, dtype=tf.float32),
                    [output_dim, 1]
                )
                
                # Project y to a scalar. 
                # (1, 36) @ (36, 1) -> (1, 1)
                # The backward gradient of this is just 'projection_vec'. 
                # No 'ZerosLike' needed!
                projected_scalars.append(tf.matmul(y, projection_vec))
        for projected_scalar in projected_scalars:
            # Calculate gradient of this scalar
            grad = tape.gradient(projected_scalar, x)
            grads.append(grad)

        # 3. Stack results
        # result shape: (1, OutputDim, 36)
        jacobian = tf.stack(grads, axis=1)
        
        return jacobian
