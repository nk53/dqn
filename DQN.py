USE_TF_KERAS = False

import tensorflow as tf
if USE_TF_KERAS:
    from tensorflow import keras
else:
    import keras

import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.losses import Huber
from keras.models import clone_model, load_model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical

import contextlib
import datetime as dt
import json
import numpy as np
import os
import pickle
import pytz
import random
import shutil
from datetime import datetime, timedelta

K.set_image_data_format('channels_first')
assert K.image_data_format() == "channels_first"

##### stringifying timedelta and datetime ####
datetime_field = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
timedelta_fields = ['days', 'microseconds', 'seconds']
def to_dict(obj):
  """convert obj (timedelta or datetime instance) to dict"""
  fields = datetime_field if isinstance(obj, datetime) else timedelta_fields
  d = {f: getattr(obj, f) for f in fields if hasattr(obj, f)}
  d["type"] =  obj.__class__.__name__#"datetime" if isinstance(obj, datetime) else "timedelta"
  return d

def from_dict(d):
  """
  Convert d (dict) to the type of object described by key "type".
  """
  tp = d.pop("type")
  obj = globals()[tp]
  return obj(**d)
#############################

def current_time():
    return dt.datetime.now(dt.timezone.utc)

eastern = pytz.timezone("US/Eastern")
def to_eastern_time(utc_time):
    return utc_time.astimezone(eastern)

@contextlib.contextmanager
def wrap_print_time(process_name):
  print ("Began {} at {}".format(process_name, to_eastern_time(current_time())))
  yield
  print("Completed {} at {}".format(process_name, to_eastern_time(current_time())))
    
class DQNmodel(object):

    def __init__(self, num_actions, backup_location, backup_frequency, use_convolutions=False, debug=False):
        self.debug=False
        self._setup(num_actions, backup_location, backup_frequency, use_convolutions)

    def set_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.updates_since_last_reset = 0
    
    def _setup(self, num_actions, backup_location, backup_frequency, use_convolutions=False):
        self.output_size = num_actions
        self.model = self.init_model(num_actions) if use_convolutions else self.init_model_no_convolutions(num_actions)
        self.target_model = clone_model(self.model)
        self.set_target_model()

        # variables for determining when to update the target model
        self.number_of_updates = 0
        self.reset_frequency = 10000

        # variables controlling exploration (tendency to pick random action)
        self.initial_exploration_rate = 1.0
        self.final_exploration_rate = 0.1
        self.annealing_steps = 10**6
        self.exploration_rate_reduction = (self.initial_exploration_rate - self.final_exploration_rate)/self.annealing_steps
        self.current_exploration_rate = self.initial_exploration_rate

        self.backup_folder = backup_location
        self.backup_every = timedelta(**backup_frequency)#hours=11) 
        self.last_backup_time = current_time()
        self.time_of_next_backup = self.calc_time_of_next_backup()
        #set_image_data_format('channels_last')

    @property
    def time_since_last_backup(self):
        return current_time() - self.last_backup_time

    def calc_time_of_next_backup(self):
        return self.last_backup_time + self.backup_every

    @staticmethod 
    def init_model_no_convolutions(num_actions, learning_rate=0.00025, momentum=0.95, added_constant=0.01):
        rmsprop = RMSprop(lr=learning_rate, rho=momentum, epsilon=added_constant)
        adam = Adam(lr=learning_rate, clipvalue=1.)
        model = Sequential()
        model.add(Dense(8, input_shape=(4,4), activation="relu"))
        model.add(Dropout(0.05))
        model.add(Flatten())
        model.add(Dense(num_actions))
        
        model.compile(optimizer=adam, loss=Huber(delta=1.0)) #or loss=mean_squared_error, or optimizer=adam?
        return model

    @staticmethod
    def init_model(num_actions, learning_rate=0.00025, momentum=0.95, added_constant=0.01):  # env.action_space.n
        
        rmsprop = RMSprop(lr=learning_rate, rho=momentum, epsilon=added_constant)
        model = Sequential()


        model.add(Conv2D(filters=32, kernel_size=(8,8), \
            input_shape=(4, 84, 84), strides=4, activation="relu",data_format="channels_first"))
        model.add(Conv2D(filters=64, kernel_size=(4,4), strides=2, activation="relu", data_format="channels_first"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation="relu", data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(num_actions))
        
        model.compile(optimizer=rmsprop, loss=Huber(delta=1.0)) #or loss=mean_squared_error, or optimizer=adam?
        return model

    @property
    def needs_reset(self):
      return self.number_of_updates == 0 or self.number_of_updates % self.reset_frequency == 0

    def _update(self, transitions, gamma=0.99, mini_batch_size = 32):
        """
        train the model
        """

        mini_batch = random.sample(transitions, mini_batch_size)
        start_states, actions, rewards, final_states, dones = map(np.array, zip(*mini_batch))

        current_scores = self.model.predict(start_states, batch_size=mini_batch_size)
        action_scores = self.target_model.predict(final_states, batch_size=mini_batch_size) 
        
        best_actions = np.argmax(action_scores, 1) # for each transition in batch, determine the best action
        best_scores = np.max(action_scores, 1)
        scores = rewards + gamma * best_scores * (dones == False)

        # overwrite the current score for the action with the target score
        # allow all other scores to remain the same
        target_scores = current_scores
        for n, a in enumerate(actions):
            target_scores[n, a] = scores[n]

        # print loss after every model reset
        verbose = self.updates_since_last_reset == 0
        self.model.fit(start_states, target_scores, verbose=verbose)
       
        # do update on reset count and reset model if necessary
        self.number_of_updates += 1
        self.updates_since_last_reset += 1 
        if self.needs_reset:
            self.set_target_model()

        # do annealing on exploration rate
        # i.e. linearly reduce current_exploration_rate from 
        # initial_exploration_rate to final_exploration_rate
        if self.current_exploration_rate > self.final_exploration_rate:
            self.current_exploration_rate = self.current_exploration_rate - self.exploration_rate_reduction
        
        if current_time() > self.time_of_next_backup:
            self.save(transitions)

    def greedy_action(self, state):
        """
        given a state (a preprocessed set of game frames/frame stack), 
        return the best action to take according to the model.
        """
        state = np.array(state)
        if state.shape == self.model.input_shape[1:]:
            state = np.array([state]) # wrap so as to be a batch of size one
        elif state.shape == self.model.input_shape:
            pass
        else:
            msg = "Was expecting state to have either {} or {}, but recieved state with shape {}."
            raise Exception(msg.format(self.model.input_shape[1:], self.model.input_shape, state.shape))

        if USE_TF_KERAS:
            state = tf.cast(state, tf.float32)
        result = self.model.predict(state)
        best_score_ind = np.argmax(result)
        return best_score_ind

    def policy(self, state):
        if self.current_exploration_rate >= random.random():
            return random.randrange(0, self.output_size)
        else:
            return self.greedy_action(state)

    def load(self, **kwargs):
        if self.debug:
            with wrap_print_time("LOAD"):
                return self._load(**kwargs)
        else:
            return self._load(**kwargs)

    def _load(self, backup_folder=None, **kwargs):

        backup_folder = backup_folder or self.backup_folder

        with open(os.path.join(backup_folder, "settings.txt"), "r") as h:
            settings = json.load(h)

        # convert back to appropriate object
        settings["backup_every"] = from_dict(settings["backup_every"])
        
        # overwrite the values in the file with current values
        settings["last_backup_time"] = current_time()
        settings["time_of_next_backup"] = self.calc_time_of_next_backup()
        
        # set up this object using settings and load models
        self.__dict__.update(**settings)
        self.__dict__.update(**kwargs)

        self.model = load_model(os.path.join(backup_folder, "model.h5"))
        self.target_model = load_model(os.path.join(backup_folder, "target_model.h5"))

        # return transitions
        replay_mem_file = os.path.join(backup_folder, "replay_memory.pkl")
        if not os.path.exists(replay_mem_file):
          raise FileNotFoundError("Cannot reload prev. training session. No replay memory file")
        return replay_mem_file


    def save(self, transitions, **kwargs):
        if self.debug:
            with wrap_print_time("SAVE"):
                return self._save(transitions, **kwargs)
        else:
            return self._save(transitions, **kwargs)


    def _save(self, transitions, backup_folder=None):
        """
        save a training session
        """
        backup_folder = backup_folder or self.backup_folder
        backup_bak = backup_folder+"bak"

        # backup the backup folder
        if os.path.isdir(backup_folder):
            shutil.copytree(backup_folder, backup_bak)

        # overwrite models in the primary backup
        self.model.save(os.path.join(backup_folder, "model.h5"))
        self.target_model.save(os.path.join(backup_folder, "target_model.h5"))
        
        #with gym.atomic_write(os.path.join(backup_folder, "replay_memory.pkl"), binary=True) as h:
        with open(os.path.join(backup_folder, "replay_memory.pkl"), "wb") as h:
          print ("replay mem size",len(transitions))
          pickle.dump(transitions, h)

        # figure out when next backup should happen, 
        # and then mark that we backup up now.
        # (calc_time_of_next_backup depends on last_backup_time)
        # we overwrite these values on load, 
        # so this is for checking the settings file while model is still running.
        self.time_of_next_backup = self.calc_time_of_next_backup()
        self.last_backup_time = current_time() 

        settings = dict(vars(self))
        settings.pop("model")
        settings.pop("target_model")
        settings["backup_every"] = to_dict(settings["backup_every"])
        settings["last_backup_time"] = to_dict(settings["last_backup_time"])
        settings["time_of_next_backup"] = to_dict(settings["time_of_next_backup"])

        # overwrite settings in primary backup
        with open(os.path.join(backup_folder, "settings.txt"), "w") as h:
          json.dump(settings, h)

        # backup complete, so previous backup can be removed
        if os.path.isdir(backup_bak):
            shutil.rmtree(backup_bak)

