# Memory Augmented Neural Networks (MANN) Repository for D-NTM

The codes to implement different types of Memory Augmented Neural Networks in [1] and more...

In this repo, you can find codes to reproduce the results on,
   1. Facebook's bAbI dataset.
   2. Permuted MNIST
   3. NTM Toy Tasks

## Some Information about the codes and the files structure
   
1. `codes/`, the general codes of the model of memory models.

    1.1. `codes/core` includes the core generic layers, mainloops, learning rules and some
    utilities to implement neural memory models safely and efficiently with Theano. One of the
    important file is:

        1.1.1. `operators.py`: includes the different operators for example a class to implement
        the similaritties between the key and the content in the memory. Or a class to implement
        the REINFORCE as `known_grads` of the `tensor.grad` function in Theano.
            * For the implementation of REINFORCE with moving averages baseline see,
                REINFORCE class.
            * For the input based baseline see,
                REINFORCEBaselineExt class.
        1.1.2. `costs.py`: this file includes different cost functions for our memory model
        implemented efficiently in theano. For example, `huber_loss` used for the input based
        baseline is implemented in this file.
            
        1.1.3. `penalty.py` different penalties and regularizers are implemented in this file. For
        example we have an alternative implementation of REINFORCE(both input based or regular
                                                                   baseline), implemented in this
        folder.

    1.2. `codes/memnet` includes the layers, mainloops and data iterators specific to implement ntm
    and dntm type of memory models. Some important files are,

        1.2.1. `addresser.py` includes different type of implementations of addressing types.
        
        1.2.2. `memory.py` implements the external memory mechanism for NTM/D-NTM models.

        1.2.3. `nmodel.py` combines different layers and implements the codes and core computation
        graph for the MANNs(using this and by changing the configs of thise file it is possible to
                            implement various types of external memory models.)

        1.2.4. `ntm_layers.py` implements the controllers and different heads(disables and enables
                                                                             some of them based on
                                                                             the options that are
                                                                             provided).

        1.2.5. `controllers.py` basic controllers for the MANN (supports either FF or
                                                               Recurrent-GRU/LSTM).

        1.2.6. `mainloop.py` implements different types of mainloop for different types of NTMs and
        the controllers.

        1.2.7. `fbBIdataiterator.py` implements the data iterator for the bAbI dataset.
        1.2.8. `babi_data_processing/` folder contains the scripts to preprocess the bAbI dataset.

2. `scripts`, the scripts to run the models and the experiments with different types of MANN
models

3. `experiments`, some of the scripts that we used for the hyperparameter search for our models.


[1] Gulcehre C, Chandar S, Cho K, Bengio Y. Dynamic Neural Turing Machine with Soft and Hard
Addressing Schemes. arXiv preprint arXiv:1607.00036. 2016 Jun 30.
