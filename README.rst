======
Deep Deterministic Policy Gradient on PyTorch
======

Overview
======
The is the implementation of `Deep Deterministic Policy Gradient <https://arxiv.org/abs/1509.02971>`_ (DDPG) using `PyTorch <https://github.com/pytorch/pytorch>`_. Part of the utilities functions such as replay buffer and random process are from `keras-rl <https://github.com/matthiasplappert/keras-rl>`_ repo. Contributes are very welcome.

Dependencies
======
* Python 3.4
* PyTorch 0.1.9 
* `OpenAI Gym <https://github.com/openai/gym>`_

Run
======
* Training : results of two environment and their training curves:

	* Pendulum-v0

	.. code-block:: console

	    $ ./main.py --debug

	.. image:: output/Pendulum-v0-run0/validate_reward.png
	    :width: 800px
	    :align: left
	    :height: 600px
	    :alt: alternate text

	* MountainCarContinuous-v0

	.. code-block:: console

	    $ ./main.py --env MountainCarContinuous-v0 --validate_episodes 100 --max_episode_length 2500 --ou_sigma 0.5 --debug

	.. image:: output/MountainCarContinuous-v0-run0/validate_reward.png
	    :width: 800px
	    :align: left
	    :height: 600px
	    :alt: alternate text

* Testing :

.. code-block:: console

    $ ./main.py --mode test --debug

TODO
======

