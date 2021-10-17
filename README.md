# Visual-Vestibular Generative Query Network

## Motivation

[Generative Query Networks](https://deepmind.com/blog/article/neural-scene-representation-and-rendering) (GQNs) learn representations of their environments by predicting what they would look like from novel viewpoints.
The architecture consists of a Representation Network that encodes a collection of viewpoint/image tuples, and a Generator Network that produces an image given the output of the Representation Network and a novel query viewpoint. 
To complete the task, the learned environment representations must be viewpoint-invariant and encode information about 3D structure/layout.

Taken as models of human scene representation, however, the model and task have 2 important shortcomings:
1. The task is not biologically plausible. Supervision is required in the form of ground-truth locations/viewpoints in an experimenter-defined reference frame.
2. The representation does not contain information about the agent and its position or orientation in the environment.

This work aims to construct a more biologically-motivated variant of GQNs that uses a similar predictive-coding task, but using vestibular movement inputs rather than ground-truth positions.
The hope is that the learned representations will contain:
- Viewpoint-invariant descriptions of the environment and its layout.
- A learned environment-dependent reference frame (i.e. emergent grid cells).
- A description of the agent's current position and orientation in the reference frame (i.e. emergent place hells and head direction cells).

## Implementation
The model consists of:
- A vision module (CNN), which takes as input the current image at time `t` and outputs a vector representing that image.
- An environment representation module (LSTM), which takes as input a sequence of visual representation and vestibular movement tuples, and whose internal state represents both the environment and the agent's location in it.
- A projection module (LSTM), which takes as input the current state of the environment and a sequence of vestibular movement tuples, and whose internal state represents both the environment and the agent's new location in it.
- A generator module (CNN), which takes as input the projection module's current state and imagines what the environment would look like from this new viewpoint.

The environment currently consists of a very simple square room to ensure that learning is possible with the current models, losses, and task description.
