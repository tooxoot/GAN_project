import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(session,
                       sample_size,
                       save_directory,
                       show,
                       save,
                       use_subplots,
                       em):
    noise_samples = em.noise_distribution(sample_size)
    train_samples = em.training_distribution(sample_size)
    generator_samples = session.run(em.generator_output, feed_dict={em.g_placeholder_name: noise_samples})

    discriminator_results_d = session.run(em.d_original_data, feed_dict={em.d_placeholder_name: train_samples})
    discriminator_results_g = session.run(em.d_original_data, feed_dict={em.d_placeholder_name: generator_samples})

    mean = (sum(discriminator_results_d) + sum(discriminator_results_g)) / (2 * sample_size)
    mean_d = sum(discriminator_results_d) / sample_size
    mean_g = sum(discriminator_results_g) / sample_size

    bool_mask_d = discriminator_results_d > mean
    bool_mask_g = discriminator_results_g > mean

    valid_train_samples = []
    invalid_train_samples = []

    valid_generator_samples = []
    invalid_generator_samples = []

    for idx in range(sample_size):
        if bool_mask_d[idx]:
            valid_train_samples.append(train_samples[idx])
        else:
            invalid_train_samples.append(train_samples[idx])

        if bool_mask_g[idx]:
            valid_generator_samples.append(generator_samples[idx])
        else:
            invalid_generator_samples.append(generator_samples[idx])

    valid_train_samples = np.array(valid_train_samples)
    invalid_train_samples = np.array(invalid_train_samples)

    valid_generator_samples = np.array(valid_generator_samples)
    invalid_generator_samples = np.array(invalid_generator_samples)

    if use_subplots:
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        scatter(axs[0], valid_train_samples, valid_generator_samples)
        scatter(axs[1], invalid_train_samples, invalid_generator_samples)
        scatter(axs[2], train_samples, generator_samples)
        show_or_save(show, save, save_directory)
        plt.close(fig)
    else:
        plot_figures(show, save, save_directory, valid_train_samples, valid_generator_samples)
        plot_figures(show, save, save_directory, invalid_train_samples, invalid_generator_samples)
        plot_figures(show, save, save_directory, train_samples, generator_samples)

    return {'mean_overal': mean[0],
            'mean_originals': mean_d[0],
            'mean_immitations': mean_g[0],
            '#valid_originals': len(valid_train_samples),
            '#invalid_originals': len(invalid_train_samples),
            '#valid_immitations': len(valid_generator_samples),
            '#invalid_immitations': len(invalid_generator_samples)}


def scatter(ax, *distributions):
    for dist in distributions:
        if len(dist) > 0:
            ax.scatter(dist[:, 0], dist[:, 1])


def show_or_save(show, save, save_directory):
    if show:
        plt.show()
    if save:
        plt.savefig(f'{save_directory}/all_samples.png', bbox_inches='tight')


def plot_figures(show, save, save_directory, *distributions):
    figure = plt.figure()
    scatter(plt, *distributions)
    show_or_save(show, save, save_directory)
    plt.close(figure)
