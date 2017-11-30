import numpy as np
import tensorflow as tf
from distribution_util import build_distribution
import model_definition as md

G_SCOPE = 'G'
D_SCOPE = 'D'
PLACEHOLDER_SCOPE = 'P'


class Execution_Model:
    noise_distribution = None
    training_distribution = None
    g_placeholder_name = None
    d_placeholder_name = None
    generator_output = None
    d_original_data = None
    d_immitated_data = None
    g_train_step = None
    d_train_step = None


def run_all(output_option):
    # Experiment 1
    noise_distribution = build_distribution(dimensions=[10])
    training_distribution = build_distribution(dimensions=[2],
                                               apply_transformation=lambda sample: np.array([sample[0] * 0.5, sample[1] * 0.25]))
    hyperparameters = {'g_placeholder_dimensions': [10],
                       'd_placeholder_dimensions': [2],
                       'g_layer_count': 1,
                       'g_neurons_per_layer': 10,
                       'g_output_units': 2,
                       'd_layer_count': 1,
                       'd_neurons_per_layer': 10,
                       'd_output_units': 1,
                       'training_loops': 1000,
                       'minibatch_size': 100,
                       'g_step_count': 1,
                       'd_step_count': 1}
    run_experiment('exp1', output_option, noise_distribution, training_distribution, hyperparameters)


def run_experiment(name,
                   output_option,
                   noise_distribution,
                   training_distribution,
                   hyperparams):
    tf.reset_default_graph()

    with tf.variable_scope(PLACEHOLDER_SCOPE):
        g_placeholder = tf.placeholder(tf.float32, [None, *hyperparams['g_placeholder_dimensions']], name='g_placeholder')
        d_placeholder = tf.placeholder(tf.float32, [None, *hyperparams['d_placeholder_dimensions']], name='d_placeholder')

    generator_arguments = [g_placeholder,                       # network_input
                           G_SCOPE,                             # scope
                           hyperparams['g_layer_count'],        # layer_count
                           hyperparams['g_neurons_per_layer'],   # neurons_per_layer
                           hyperparams['g_output_units'],       # output_units
                           tf.nn.relu]                          # tf_activation_function

    discriminator_arguments = [d_placeholder,                       # network_input
                               D_SCOPE,                             # scope
                               hyperparams['d_layer_count'],        # layer_count
                               hyperparams['d_neurons_per_layer'],  # neurons_per_layer
                               hyperparams['d_output_units'],       # output_units
                               tf.nn.relu]                          # tf_activation_function

    training_model = md.build_training_model(generator_arguments, discriminator_arguments)

    # print(*tf.global_variables(), sep='\n')

    print(*tf.trainable_variables(), sep='\n')

    em = Execution_Model()
    em.noise_distribution = noise_distribution
    em.training_distribution = training_distribution
    em.g_placeholder_name = g_placeholder.name
    em.d_placeholder_name = d_placeholder.name
    em.generator_output    = training_model[0] # noqa
    em.d_original_data     = training_model[1] # noqa
    em.d_immitated_data    = training_model[2] # noqa
    em.g_train_step        = training_model[3] # noqa
    em.d_train_step        = training_model[4] # noqa

    md.train_model(output_option,                   # output_option
                   name,                            # session_name
                   hyperparams['training_loops'],   # training_loops
                   hyperparams['minibatch_size'],   # minibatch_size
                   hyperparams['g_step_count'],     # g_step_count
                   hyperparams['d_step_count'],     # d_step_count
                   em)                              # execution_model


if __name__ == "__main__":
    output_option = {'step_count': 10,
                     'save_session': True,
                     'print_diagramms': True,
                     'sample_size': 100,
                     'show': False,
                     'save': True,
                     'use_subplots': False}
    run_all(output_option)
