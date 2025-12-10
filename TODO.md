# TODO

## TensorFlow to Onnx to OpenCV troubles

Try the following:

- https://onnxruntime.ai/docs/tutorials/tf-get-started.html
- https://github.com/lutzroeder/netron/issues/71
- Maybe try `model.export(...)` instead of `tf.saved_model.save(...)`?

## Not Important But Maybe Useful

- Having "menus" in matplotlib plots: https://matplotlib.org/stable/gallery/widgets/menu.html

## Possibly/likely _after_ paper submission

- Include gravity direction as another input vector.
- Try a NN with 10 (or so) output nodes that interpolates between the different
  classes of predictions.
- Prediction scheme that finds the point x in local space that moved the least
  over the last N frames and assumes it will be fixed in space again this frame.
  Similar to circular movement, but one can quickly think of examples where the
  results would differ.
- Tailor normalization for each column by graphing histograms and referring to https://developers.google.com/machine-learning/crash-course/numerical-data/normalization
- Concat a sliding-window fully-connected NN (FCNN) or RNN with my "tabular"
FCNN so that you get the precomputed columns _plus_ any new ones the sliding
window one computes.


## Some Things I Tried That Did Not Work

This is not exhaustive, since (a) some of them I "documented" by just leaving
the code for them present but uncalled, and (b) other were small things I did
not consider documenting until now.

- Scaling all non-OneHot columns with RobustScaler, PowerTransformer, and
MinMaxScaler instead of StandardScaler. This ties into the TODO above of using
different ones for different columns.

