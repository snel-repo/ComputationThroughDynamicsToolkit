import pytorch_lightning as pl
import torch
from gymnasium import Env
from torch import nn


class TaskTrainedWrapper(pl.LightningModule):
    """Wrapper for a task trained model

    Handles the training and validation steps for a task trained model.

    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        input_size=None,
        output_size=None,
        task_env: Env = None,
        model: nn.Module = None,
    ):
        """Initialize the wrapper

        args:
            learning_rate (float):
                The learning rate for the optimizer
            weight_decay (float):
                The weight decay for the optimizer
            input_size (int):
                The size of the input to the model
                ( NBFF: N pulsatile inputs
                  MultiTask: 20 total: 5 time-varying inputs and 15 task flag inputs
                  RandomTarget: 17 total: 12 muscle inputs,
                    2 visual inputs, 2 target inputs, 1 go cue)
            output_size (int):
                The size of the output of the model
                ( NBFF: N total: The state of the environment
                  MultiTask: 3 total: 2 outputs and 1 fixation
                RandomTarget: 6 muscle activation commands)
            task_env (Env):
                The environment to learn to operate in
            model (nn.Module):
                torch.nn.Module dynamics model that transforms the inputs to the outputs
            loss_func (LossFunc):
                The loss function to use
                - see loss_func.py for examples

        """
        super().__init__()
        self.save_hyperparameters()
        self.task_env = task_env
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def set_environment(self, task_env: Env):
        """Set the environment for the training pipeline
        This method sizes the input and output of the model based on the environment
        and checks to see whether the environment is coupled or if noise is dynamic"""

        self.task_env = task_env
        self.input_size = (
            task_env.context_inputs.shape[0] + task_env.observation_space.shape[0]
        )
        self.output_size = task_env.action_space.shape[0]
        self.loss_func = task_env.loss_func
        if hasattr(task_env, "state_label"):
            self.state_label = task_env.state_label
        if hasattr(task_env, "dynamic_noise"):
            self.dynamic_noise = task_env.dynamic_noise
        else:
            self.dynamic_noise = 0.0

    def set_model(self, model: nn.Module):
        """Set the model for the training pipeline"""
        self.model = model
        self.latent_size = model.latent_size

    def configure_optimizers(self):
        """Configure the optimizer"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(self, ics, inputs, inputs_to_env=None, stim_inputs=None):
        """Pass data through the model
        args:
            ics (torch.Tensor):
                The initial conditions for the environment
            inputs (torch.Tensor):
                The pre-computed inputs to the model
                All NBFF and MultiTask inputs are pre-computed,
                Only the target location and go cue are pre-computed in RandomTarget
            targets (torch.Tensor):
                The targets for the model
                (i.e., the outputs of the environment or MotorNet goal)

        returns:
            output_dict (dict):
                A dictionary containing the controlled variable,
                    latent activity, actions, states, and joints
                controlled: The variable being optimized by the model
                    NBFF: The "memorized" bit state
                    MultiTask: The output and fixation signals
                    RandomTarget: The fingertip position
                latents: The latent activity of the model
                actions: The actions taken by the model
                    NBFF: The "memorized" bit state (same as controlled)
                    MultiTask: The output and fixation signals (same as controlled)
                    RandomTarget: The muscle activation commands
                states: The state of the environment that the model is interacting with
                    NBFF: N/A
                    MultiTask: N/A
                    RandomTarget: Kinematics of the arm
                joints: The joint angles of the arm (only for coupled environments)


        """
        batch_size = ics.shape[0]

        # Step 1: For environments with coupled dynamics,
        # reset the environment with the initial conditions
        if self.task_env.coupled_env:
            options = {"ic_state": ics}
            env_states, info = self.task_env.reset(
                batch_size=batch_size, options=options
            )
            env_state_list = []
            joints = []  # Joint angles
        else:
            env_states = None
            env_state_list = None

        # Step 2: If the model has functionality to reset the hidden state, do so
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(batch_size=batch_size).to(self.device)
        elif hasattr(self.model, "init_hidden_from_ic"):
            hidden = self.model.init_hidden_from_ic(ics).to(self.device)
        else:
            hidden = torch.zeros(batch_size, self.latent_size).to(self.device)

        # Step 3: Step through each trial over time, providing inputs to the model
        latents = []  # Latent activity of TT model
        controlled = []  # Variable controlled by model
        actions = []  # Actions taken by model (sometimes = controlled)

        count = 0
        terminated = False
        while not terminated and len(controlled) < self.task_env.n_timesteps:
            # Step 3a: For coupled environments,
            # add the environment state to the model input
            if self.task_env.coupled_env:
                model_input = torch.hstack((env_states, inputs[:, count, :]))
            else:
                model_input = inputs[:, count, :]

            # Step 3b: If needed, add dynamic noise to the model input
            if self.dynamic_noise > 0:
                model_input = (
                    model_input + torch.randn_like(model_input) * self.dynamic_noise
                )

            # Step 3c: Pass the inputs into the model
            # and get the "action" and new hidden state
            action, hidden = self.model(model_input, hidden)

            # Step 3d: If coupled, apply action to environment using step()
            if self.task_env.coupled_env:
                # Apply external loads (if they exist)
                if inputs_to_env is not None:
                    env_states, _, terminated, _, info = self.task_env.step(
                        action=action,
                        inputs=inputs[:, count, :],
                        endpoint_load=inputs_to_env[:, count, :],
                    )
                else:
                    env_states, _, terminated, _, info = self.task_env.step(
                        action=action, inputs=inputs[:, count, :]
                    )
                if stim_inputs is not None:
                    hidden += stim_inputs[:, count, :]
                controlled.append(info["states"][self.state_label])
                joints.append(info["states"]["joint"])
                actions.append(action)
                env_state_list.append(env_states)
            else:
                controlled.append(action)
                actions.append(action)

            latents.append(hidden)
            count += 1

        # Step 4: Compile outputs and return:
        controlled = torch.stack(controlled, dim=1)
        latents = torch.stack(latents, dim=1)
        actions = torch.stack(actions, dim=1)
        if self.task_env.coupled_env:
            states = torch.stack(env_state_list, dim=1)
            joints = torch.stack(joints, dim=1)
        else:
            states = None
            joints = None

        output_dict = {
            "controlled": controlled,  # BETTER VARIABLE NAME?
            "latents": latents,
            "actions": actions,
            "states": states,
            "joints": joints,
        }
        return output_dict

    def training_step(self, batch, batch_ix):
        """Training step for the model
        args:
            batch (tuple):
                The batch of data to train on
                ics: The initial conditions for the environment
                inputs: The pre-computed inputs to the model
                targets: The targets for the model
                conds: The conditions for the model
                extras: Extra information for the model
                inputs_to_env: The inputs to the environment
            batch_ix (int):
                The index of the batch

        returns:
            loss_all (torch.Tensor):
                The loss for the batch

        """
        # Step 1: Unpack the batch
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]
        inputs_to_env = batch[6]

        # Step 2: Pass data through the model
        output_dict = self.forward(ics, inputs, inputs_to_env)

        # Step 3: Compute the weighted loss using the environments loss function
        loss_dict = {
            "controlled": output_dict["controlled"],
            "latents": output_dict["latents"],
            "actions": output_dict["actions"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            "epoch": self.current_epoch,
        }
        loss_all = self.loss_func(loss_dict)
        # Step 4: If the model itself has has a loss function, add its contribution
        if hasattr(self.model, "model_loss"):
            model_loss = self.model.model_loss(loss_dict)
            self.log("train/model_loss", model_loss)
            loss_all += model_loss
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        """Validation step for the model
        args:
            batch (tuple):
                The batch of data to train on
                ics: The initial conditions for the environment
                inputs: The pre-computed inputs to the model
                targets: The targets for the model
                conds: The conditions for the model
                extras: Extra information for the model
                inputs_to_env: The inputs to the environment
            batch_ix (int):
                The index of the batch

        returns:
            loss_all (torch.Tensor):
                The loss for the batch

        """

        # Step 1: Unpack the batch
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]
        inputs_to_env = batch[6]

        # Step 2: Pass data through the model
        output_dict = self.forward(ics, inputs, inputs_to_env=inputs_to_env)

        # Step 3: Compute the weighted loss
        loss_dict = {
            "controlled": output_dict["controlled"],
            "actions": output_dict["actions"],
            "latents": output_dict["latents"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            "epoch": self.current_epoch,
        }
        loss_all = self.loss_func(loss_dict)

        # Step 4: Compute the loss using the loss function object
        if hasattr(self.model, "model_loss"):
            model_loss = self.model.model_loss(loss_dict)
            self.log("valid/model_loss", model_loss)
            loss_all += model_loss
        self.log("valid/loss", loss_all)
        return loss_all
