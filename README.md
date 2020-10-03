# BERT-TF2
BERT reimplentation on Tensorflow v2.3.0 

Work in progess..
> BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
>
>[Paper](https://arxiv.org/abs/1810.04805) |
>[Official code](https://github.com/google-research/bert) |
>[TF/models code](https://github.com/tensorflow/models/tree/fcd690b14b04c11e7f25b9d473db056c4b7947b3/official/nlp/bert)

### Plans
- BERT model on TF 2.x Keras API (done)
- Loading official BERT weights 
- Data parallelism for GPU, TPU
- Should all the data be prepared as TFRecord before training? Will preparing them on training be a unbearable bottleneck?
- Any TPU-ineffecient operation? (e.g. reshaping tensors) [link](https://cloud.google.com/tpu/docs/performance-guide?hl=ko)
- Training ELECTRA based on this model
- Practice model parallelism (Tensorflow mesh? [link](https://github.com/tensorflow/mesh)) (or maybe later.. tf.distribute [doc link](https://www.tensorflow.org/api_docs/python/tf/distribute?hl=en))
> **tf.distribute doc**
>
> Data parallelism is where we run multiple copies of the model on different slices of the input data. This is in contrast to model parallelism where we divide up a single copy of a model across multiple devices. Note: we only support data parallelism for now, **but hope to add support for model parallelism in the future.**

### Contact 
Han Seungho (danhahn61@gmail.com)