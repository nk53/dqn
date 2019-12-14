import gym
from collections import deque
import numpy as np
import os
from gym import wrappers
import json

class RollingMeanReturn(wrappers.RecordEpisodeStatistics):
    """
    writes
    calcs rolling mean in real time and writes to graph
    """
    def __init__(self, env, window=100):
        """
        window: number of returns to average over
        appendable: an instance of an object with append
        """
        super(RollingMeanReturn, self).__init__(env, deque_size=window)
        self.window = window
    

    @property
    def average(self):
        return np.average(np.array(self.return_queue))        


    def step(self, action):
        observation, reward, done, info = super(RollingMeanReturn, self).step(action)

        # only take new average at end of episode if the window is full.
        if done and len(self.return_queue) == self.window:
            info['episode']["a"] = self.average

        return observation, reward, done, info


class RecordInfo(gym.Wrapper):
    def __init__(self, env, filepath, info_tags=None, overwritefile=False):
        """
        info_tags (list): 
            if None, save all elements of info
            save only elements of info with these keys (if they are present.)
        """
        super(RecordInfo, self).__init__(env)
        self.path = filepath
        self.tags = info_tags # the names of labels in info

        # try to init the empty file
        if os.path.exists(self.path) and not overwritefile:
            raise Exception("File exists. Delete or set overwritefile to True.")
        open(self.path, "w").close() # erase the file


    def step(self, action):
        observation, reward, done, info = super(RecordInfo, self).step(action)

        tags_to_save = self.tags if self.tags else info.keys()
        d = {t: info[t] for t in tags_to_save if t in info}
        if d:
            with open(self.path, "a") as h:
                json.dump(d,h)
                h.write(os.linesep)
        return observation, reward, done, info

            
class AppendableFile(object):
    def __init__(self, path):
        self.fname = path
        
    def clear(self):
        """blank the file."""
        open(self.fname, "w").close() 

    def append(self, value):
        """
        add value to end of file.
        """
        with open(self.fname, "a") as h:
            h.write(value)

if __name__ == "__main__":
    calc_rolling_avg = True
    rolling_avgs_file = "rolling_avgs.txt"

    env_name = 'CartPole-v0' #'StarGunnerNoFrameskip-v4'
    
    # setup
    env = gym.make(env_name)

    if 'StarGunner' in env_name:
        env = gym.wrappers.EpisodicLifeEnv(env)
        env = gym.wrappers.GrayScaleObservation(env) 

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStack(env, 4, lz4_compress=True)

    if calc_rolling_avg:
        env = RollingMeanReturn(env, window=100)
        env = RecordInfo(env, rolling_avgs_file, ["a"], overwritefile=True)
