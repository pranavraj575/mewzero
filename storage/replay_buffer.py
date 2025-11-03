"""
from werner-duvaud/muzero-general

config:
tensor_tuple -> whether all data will be a tuple of tensors. If true, will sample and batch automatically
capacity -> max capacity of buffer

"""
import copy
import time

import numpy as np
import torch
from collections import deque

import torch, os, shutil, pickle


class AbsReplayBuffer:
    def __init__(self, config):
        self.config = config

    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def clear(self):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def sample(self, batch=None, **kwargs):
        if batch is None:
            batch = self.config.get('batch', 128)
        if self.config.get('tensor_tuple', False):
            stuff = [self.sample_one() for _ in range(batch)]
            # stuff=[(t11,t12,...,t1n),(t21,t22,...,t2n),...]
            # convert this to stack((t11,t21,t31,...)), stack((t12,t22,t32,...)), ...
            return tuple(torch.stack([s[i] for s in stuff]) for i in range(len(stuff[0])))
        else:
            return [self.sample_one() for _ in range(batch)]

    def __len__(self):
        raise NotImplementedError


class ReplayBufferList(AbsReplayBuffer):
    def __init__(self, config, initial_buffer=None):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.get('capacity', int(1e6)))
        if initial_buffer is not None:
            for item in initial_buffer:
                self.push(item)

    def clear(self):
        super().clear()
        self.buffer = deque(maxlen=self.config.get('capacity', int(1e6)))

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        pickle.dump(self.buffer, open(os.path.join(save_dir, 'buffer.pkl'), 'wb'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.buffer = pickle.load(open(os.path.join(save_dir, 'buffer.pkl'), 'rb'))

    def reset_storage(self):
        self.clear()

    def push(self, item):
        if self.buffer.maxlen == len(self.buffer):
            disp = self.buffer[0]
        else:
            disp = None
        self.buffer.append(item)
        return disp

    def _grab_item_by_idx(self, idx):
        return self.buffer[idx]

    def sample_one(self):
        return self.buffer[torch.randint(0, self.__len__(), (1,))]

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError
        return self._grab_item_by_idx(idx=item)

    def __len__(self):
        return len(self.buffer)


class ReplayBufferDiskStorage(AbsReplayBuffer):
    def __init__(self,
                 storage_dir=None,
                 capacity=1e6,
                 device=None,
                 ):
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.device = device
        if storage_dir is not None:
            self.set_storage_dir(storage_dir=storage_dir)

    def clear(self):
        super().clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(src=self.storage_dir, dst=save_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.clear()
        shutil.copytree(src=save_dir, dst=self.storage_dir)
        self.load_place(force=False)

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.reset_storage()

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0
        self.save_place()

    def save_place(self):
        """
        saves idx and size to files as well
        """
        pickle.dump(
            {
                'size': self.size,
                'idx': self.idx,
            },
            open(self._get_file('info'), 'wb')
        )

    def load_place(self, force=False):
        info_file = self._get_file(name='info')
        if os.path.exists(info_file):
            dic = pickle.load(open(info_file, 'rb'))
            self.size = dic['size']
            self.idx = dic['idx']
        else:
            if force:
                print('failed to load file:', info_file)
                print('resetting storage')
                self.reset_storage()
            else:
                raise Exception('failed to load file: ' + info_file)

    def _get_file(self, name):
        return os.path.join(self.storage_dir, str(name) + '.pkl')

    def push(self, item):
        if self.size == self.capacity:
            disp = self.__getitem__(self.idx)
        else:
            disp = None
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1)%self.capacity)

        self.save_place()
        return disp

    def _grab_item_by_idx(self, idx, change_device=True):
        item = pickle.load(open(self._get_file(name=idx), 'rb'))
        return self._convert_device(item=item, change_device=change_device)

    def _convert_device(self, item, change_device):
        if change_device:
            if type(item) == tuple:
                item = tuple(self._convert_device(t, change_device=change_device)
                             for t in item)
            elif torch.is_tensor(item):
                item = item.to(self.device)
        return item

    def sample_one(self):
        return self[torch.randint(0, self.size, (1,))]

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self._grab_item_by_idx(idx=int((self.idx + item)%self.size))

    def __len__(self):
        return self.size


if __name__ == '__main__':
    test = ReplayBufferDiskStorage(capacity=3, storage_dir=os.path.join('replay_buffer_test'))
    test.extend('help')
    print([test[i] for i in range(len(test))])
    test.clear()


class ReplayBuffer(AbsReplayBuffer):
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config, initial_buffer=None):
        super().__init__(config, initial_buffer)
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                            np.abs(
                                root_value - self.compute_target_value(game_history, i)
                            )
                            **self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = np.array(priorities, dtype="float32")
                game_history.game_priority = np.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
                self.config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                *len(actions)
            )
            if self.config.PER:
                weight_batch.append(1/(self.total_samples*game_prob*pos_prob))

        if self.config.PER:
            weight_batch = np.array(weight_batch, dtype="float32")/max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = np.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = np.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = np.array(game_probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = np.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = np.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities/sum(game_history.priorities)
            position_index = np.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = np.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                                                                         : end_index - start_index
                                                                         ]

                # Update game priorities
                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                   == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value*self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
                game_history.reward_history[index + 1: bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                         reward
                         if game_history.to_play_history[index]
                            == game_history.to_play_history[index + i]
                         else -reward
                     )*self.config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
                state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1/len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1/len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(np.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions
