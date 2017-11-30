import tensorflow as tf
import print_diagramms as pd
import os


def build_fully_connected_network(network_input,
                                  scope,
                                  layer_count=1,
                                  neurons_per_layer=10,
                                  output_units=1,
                                  tf_activation_function=tf.nn.relu,
                                  reuse_weights=False):
    with tf.variable_scope(scope):
        if reuse_weights:
            tf.get_variable_scope().reuse_variables()

        current_input = network_input

        for layer_idx in range(layer_count):
            current_layer = tf.layers.dense(inputs=current_input, units=neurons_per_layer, activation=tf_activation_function, name=f'layer_{layer_idx}')
            current_input = current_layer

        output_layer = tf.layers.dense(inputs=current_input, units=output_units, name='scope_out')
        sigmoid = tf.nn.sigmoid(output_layer)

        return output_layer, sigmoid


def build_training_model(generator_arguments,
                         discriminator_arguments):
    _, generator_output = build_fully_connected_network(*generator_arguments, reuse_weights=False)
    _, d_original_data = build_fully_connected_network(*discriminator_arguments, reuse_weights=False)
    _, d_immitated_data = build_fully_connected_network(generator_output, *discriminator_arguments[1:], reuse_weights=True)

    loss_originals = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_original_data,
                    labels=tf.ones_like(d_original_data)
                  ))
    loss_immitions = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_immitated_data,
                    labels=tf.zeros_like(d_immitated_data)
                  ))
    d_loss = loss_originals + loss_immitions

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_immitated_data,
                labels=tf.ones_like(d_immitated_data)
             ))

    tf_vars = tf.trainable_variables()

    g_vars = [var for var in tf_vars if generator_arguments[1] in var.name]
    d_vars = [var for var in tf_vars if discriminator_arguments[1] in var.name]

    d_train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss, var_list=d_vars)
    g_train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_loss, var_list=g_vars)

    return generator_output, d_original_data, d_immitated_data, g_train_step, d_train_step


def train_model(output_option,
                session_name,
                training_loops,
                minibatch_size,
                g_step_count,
                d_step_count,
                em):
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    for current_step in range(training_loops):
        for k in range(d_step_count):
            noise_batch = em.noise_distribution(minibatch_size)
            train_batch = em.training_distribution(minibatch_size)
            session.run(em.d_train_step, feed_dict={em.g_placeholder_name: noise_batch, em.d_placeholder_name: train_batch})

        for l in range(g_step_count):
            noise_batch = em.noise_distribution(minibatch_size)
            session.run(em.g_train_step, feed_dict={em.g_placeholder_name: noise_batch})

        if not output_option or current_step % output_option['step_count'] != 0:
            continue

        directory = f'/tmp/{session_name}/step_{current_step}'

        if not os.path.exists(directory):
            os.makedirs(directory)

        if output_option['save_session']:
            saver = tf.train.Saver()
            saver.save(session, f'{directory}/model.ckpt')

        if output_option['print_diagramms']:
            evaluation_results = pd.plot_distributions(session,                         # session
                                                       output_option['sample_size'],    # sample_size
                                                       directory,                       # save_directory
                                                       output_option['show'],           # show
                                                       output_option['save'],           # save
                                                       output_option['use_subplots'],   # use_subplots
                                                       em)                              # em
            if output_option['show']:
                print(str(evaluation_results))
            if output_option['save']:
                f = open(f'{directory}/evaluation.log', 'w+')
                f.write(str(evaluation_results))
                f.close()
