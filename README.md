# attention is all you need: transformer implementation

## the paper that changed everything

in 2017, vaswani and colleagues published *attention is all you need*. it introduced the transformer, a model that replaced recurrence and convolutions with attention alone. this was a turning point in machine translation and later in nlp as a whole. the architecture was faster, easier to parallelize, and achieved better results. from there, we got bert, gpt, and the large language models we rely on today.

## my implementation journey

this was my first full implementation of the transformer model from scratch. before this, i had worked with the theory a lot and also spent time fixing and adjusting implementations inside huggingface’s transformers library. but doing the whole thing myself, piece by piece, really made the concepts stick.

for training, i went with english to french translation. it felt close to the original paper’s spirit while also giving me a practical way to test the model end to end.

## building the model

i followed the original design but wrote everything out in pytorch. the code is modular: embeddings, positional encodings, multi-head attention, feed-forward networks, encoder/decoder blocks, and finally the projection layer.

**multi-head attention**
implemented scaled dot-product attention:

```
attention(q, k, v) = softmax(qk^t / sqrt(d_k))v
```

and then extended it to multiple heads, each learning different relationships in the sequence.

**positional encoding**
since there’s no recurrence, the model needs positional information. i used the sinusoidal formulation from the paper. my first version followed it directly, but i quickly ran into the numerical stability issue for larger dimensions. i fixed it later by switching to the log-based version.

**encoder and decoder**

* the encoder has stacked layers of self-attention and feed-forward networks, each wrapped with residual connections and layer norm.
* the decoder is similar but includes masked self-attention and encoder-decoder attention. the masking was important to stop the decoder from peeking ahead at future tokens.

**projection layer**
a simple linear + log softmax to map the decoder outputs back to vocabulary space.

## training setup

* dataset: english to french (opus_books)
* tokenization: word-level with [sos], [eos], [pad], and [unk] tokens
* sequence length: padded to 350 tokens
* batch size: 64
* model parameters:

  * d_model = 512
  * d_ff = 2048
  * heads = 8
  * layers = 6 (encoder and decoder)
  * dropout = 0.1
  * label smoothing = 0.1

optimization followed the paper closely: adam with β1 = 0.9, β2 = 0.98, ε = 1e-9 and the custom learning rate schedule with warmup steps (4000).

## observations while implementing

* **numerical stability** positional encoding in its raw form can overflow, switching to a log formulation avoids that.
* **parallelism** reshaping tensors for multi-head attention is far more efficient than looping.
* **gradient stability** residual connections plus layer norm are not optional. without them, training was unstable.
* **attention patterns** the model learned local patterns in lower layers and global dependencies in higher ones, very much in line with the paper.

## what i learned

1. attention really does hold up as the core operation, no recurrence needed.
2. initialization (like xavier uniform) makes a real difference when combined with residuals.
3. the modularity of the architecture makes it clean to implement and extend.
4. sinusoidal positional encoding works better than i expected, and it gives the model relative position awareness almost for free.

this implementation ended up being a faithful reproduction of the transformer. writing it from scratch made me see every moving part clearly, from the shape of tensors in attention all the way to the training dynamics. it’s one thing to import from huggingface, and another to know exactly how the pieces fit together and why.

## running training

to start training, just move into the training directory and run:

```bash
cd train && python train.py
```

## environment setup with docker

i set up the environment with docker so everything runs consistently. here are the commands i use:

```bash
# build the docker image
docker build -t transformer .

# run the container
docker run -it --rm transformer
```

if you want to mount your local project and use gpu acceleration, you can run it like this:

```bash
docker run -it --rm \
  -v $(pwd):/app \
  --gpus all \
  transformer
```