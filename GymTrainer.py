import numpy as np
import pickle
import os
from collections import deque
from random import random

def luminance(frame):
    """Maps each pixel to its luminance value.

    Expected input has dimensions (..., 3)
    """
    return np.dot(frame, np.array([.3, .59, .11]))

def operate_on_frameset(transition, callback):
    """Returns a new transition with only the frameset elements changed
    by the callback function

    framesets are elements 0 and 3
    """
    transition = list(transition)
    transition[0] = callback(transition[0])
    transition[3] = callback(transition[3])
    return tuple(transition)

def reduce_dims(frameset):
    """Somehow either the FrameStack or ResizeObservation gym wrappers
    is causing the (84,84) frame I specified to be stored as (84,84,1)

    Solution is simple, unpack each frame as shown below. LazyFrames takes
    care of decompressing each frame, so it's OK
    """
    return [frame[:,:,0] for frame in frameset]

def reduce_sample_dims(transitions):
    """Takes a sample of multiple transitions, and returns a version with
    reduced frame dims"""
    return [operate_on_frameset(tx, reduce_dims) for tx in transitions]

class GymTrainer:
    def __init__(self, env, preprocess_callbacks=(), replay_memory_size=0,
            agent_history_size=4, action_repeat=4, pkfile=None, logfile=None):
        """Setup a model for training

        Parameters
        ==========
            env                   training environment, e.g.,
                                  gym.make('env_name')
            preprocess_callbacks  function or list of functions
            agent_history         num frames in each replay memory
            action_repeat         number of times to repeat last action
                                  before choosing new action
        """
        # function(s) to apply to each frame before passing to model
        self.preprocess_frame = self.get_frame_preprocessor(preprocess_callbacks)

        self.env = env

        self.action_repeat = action_repeat

        if pkfile != None and os.path.isfile(pkfile):
          with open(pkfile, 'rb') as fh:
            self.replay_memory = pickle.load(fh)
            assert isinstance(self.replay_memory, deque), pkfile+" not a valid replay memory file"
        else:
          self.replay_memory = deque((), maxlen=replay_memory_size)

        if logfile != None:
            if os.path.isfile(logfile):
                self.logfile = open(logfile, 'a')
            else:
                self.logfile = open(logfile, 'w')
        else:
            self.logfile = None
            print("Continuing without logging scores")

    def train(self, policy='random', observe=False, num_steps=1, verbose=False,
            render=False):
        """Train the given model by playing the game using the given policy

        The value returned is the per-episode sum of the actual values received

        `policy` can be either the string 'random' (default), or a function of
        the form `policy(frames)` taking a sequence of frames of length
        `self.agent_history` and returning one of the actions in env.action_space

        If not False, `observe` should be a callable taking a tuple of the
        form (before_frames, action, reward, after_frames, is_done)

        `batch_size` is the number of observations between calls to observe,
        or, if 0, observation is skipped

        If `verbose` is True, the values of step_num, episode_step, and reward
        will be printed after each reward. At the end of each episode, the score
        will also be included

        If `render` is True, each frame will be rendered as it is generated. Note
        that this is very slow
        """
        if policy == 'random':
            policy = self.random_action
        else:
            if not callable(policy):
                errmsg = 'unrecognized policy function type: {}'
                errmsg = errmsg.format(type(policy).__name__)
                raise TypeError(errmsg)

        if int(num_steps) == num_steps:
            num_steps = int(num_steps)
        else:
            errmsg = 'Step number should be an integer, got: {}'
            errmsg.format(step_num)
            raise TypeError(errmsg)

        # local variable aliases
        action_repeat = self.action_repeat
        preprocess_frame = self.preprocess_frame

        observation = self.env.reset()
        self.agent_history = observation
        action = policy(self.agent_history)
        reward = 0

        # step number *within* an episode
        episode_step = 0
        # total score *within* an episode
        score = 0

        # step_num: total steps seen across all episodes
        for step_num in range(num_steps):
            if render:
                self.env.render()

            # actions chosen once every action_repeat frames
            if episode_step and episode_step % action_repeat == 0:
                prev = self.agent_history
                curr = observation

                # the actual value stored in self.replay_memory
                transition = (prev, action, reward, curr, done)

                self.replay_memory.append(transition)
                if callable(observe):
                    observe(self.replay_memory)

                if verbose and reward:
                    print(step_num, episode_step, reward)

                # save transition, clear frames
                self.agent_history = observation

                # get a new action according to the given policy
                action = policy(self.agent_history)
                reward = 0

            observation, new_reward, done, info = self.env.step(action)

            # keep track of the actual score before clipping it
            score += new_reward

            # if we got a reward within last episode_step frames, use it;
            # else, clip rewards to (0, 1)
            reward = reward or (new_reward and 1 or 0)

            episode_step += 1

            if done:
                if verbose:
                    print(step_num, episode_step, reward, score)

                if self.logfile:
                    self.logfile.write("{}\n".format(score))
                score = 0
                episode_step = 0

                prev = self.agent_history
                curr = observation

                # the actual value stored in self.replay_memory
                transition = (prev, action, reward, curr, done)

                self.replay_memory.append(transition)
                if callable(observe):
                    observe(self.replay_memory)

                agent_history = []

                action = policy(self.agent_history)
                reward = 0
                observation = self.env.reset()
                self.agent_history = observation

    def get_frame_preprocessor(self, callbacks=()):
        """Returns a function that preprocesses frames

        Preprocessing functions (callbacks) should either be a function or a
        sequence of functions. Callbacks are called in the order supplied,
        should take a single argument (frame), and should return a processed
        frame
        """
        if '__iter__' in dir(callbacks):
            for func in callbacks:
                if not callable(func):
                    errmsg = 'unrecognized callback type: {}'
                    errmsg = errmsg.format(type(callbacks).__name__)
                    raise TypeError(errmsg)
            def preprocess_frame(frame):
                for func in callbacks:
                    frame = func(frame)
                return frame
            return preprocess_frame
        elif callable(callbacks):
            return callbacks
        else:
            errmsg = 'unrecognized callback type: {}'
            errmsg = errmsg.format(type(callbacks).__name__)
            raise TypeError(errmsg)

    def random_action(self, *args, **kwargs):
        """Returns an action sampled uniformly from the action space

        Passed arguments are ignored
        """
        return self.env.action_space.sample()

if __name__ == '__main__':
    import gym
    # basic test of functionality
    env = gym.make('StarGunnerNoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4,
            grayscale_obs=False, screen_size=84)

    trainer = GymTrainer(env, preprocess_callbacks=luminance, replay_memory_size=4,
            agent_history_size=4, action_repeat=4)

    trainer.train(num_steps=10000, verbose=True, render=True)

    print("Experienced", len(scores), "episodes and received the following scores:")
    print(', '.join(map(str, scores)))
