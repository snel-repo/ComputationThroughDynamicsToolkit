import numpy as np
import pytorch_lightning as pl
import torch

from ctd.comparison.utils import FixedPoints


def find_fixed_points(
    model: pl.LightningModule,
    state_trajs: np.array,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if len(inputs.shape) > 1:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)

    # Select the initial states
    states = state_pts[idx]
    if len(inputs.shape) > 1:
        inputs = inputs[idx]
    else:
        inputs = inputs.unsqueeze(0).repeat(n_inits, 1)

    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)

    # Require gradients for the states
    states = states.detach()
    initial_states = states.detach().cpu().numpy()
    states.requires_grad = True

    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)

    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    q_history = {
        "iter": [],
        "mean": [],
        "median": [],
        "p10": [],
        "p90": [],
    }
    history_every = max(1, min(50, max_iters // 200 if max_iters >= 200 else 1))

    def record_q_history(iteration, q_values):
        q_history["iter"].append(int(iteration))
        q_history["mean"].append(float(np.mean(q_values)))
        q_history["median"].append(float(np.median(q_values)))
        q_history["p10"].append(float(np.percentile(q_values, 10)))
        q_history["p90"].append(float(np.percentile(q_values, 90)))

    while True:
        # Compute q and dq for the current states
        F = model(inputs, states)
        q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        if iter_count == 1 or iter_count % history_every == 0:
            record_q_history(iter_count, q_np)

        # Report progress (in-place; overwrites previous line)
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(
                f"\r  iter {iter_count}/{max_iters}  "
                f"q={mean_q:.2E}+/-{std_q:.2E}  "
                f"dq={mean_dq:.2E}+/-{std_dq:.2E}",
                end="",
                flush=True,
            )

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("\n  Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        iter_count += 1
    # Collect fixed points

    qstar = q.cpu().detach().numpy()
    if not q_history["iter"] or q_history["iter"][-1] != iter_count:
        record_q_history(iter_count, qstar)

    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )
    all_fps.q_history = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in q_history.items()
    }

    print(f"\nFound {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, inputs, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps


def find_fixed_points_coupled(
    model: pl.LightningModule,
    context_inputs: np.array,
    env_states: np.array,
    model_states: np.array,
    joint_states: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model_states = model_states.to(device)
    env_states = env_states.to(device)
    context_inputs = context_inputs.to(device)
    joint_states = joint_states.to(device)

    # Model takes in "model_input" and "hidden"
    # Model input is the concatenation of
    # the environment states and the context inputs (in that order)
    # Hidden is the hidden state of the model

    rand_inds = torch.randint(0, env_states.size(0), (n_inits,), device=device)
    env_states = env_states[rand_inds]
    model_states = model_states[rand_inds]
    context_inputs = context_inputs[rand_inds]
    joint_states = joint_states[rand_inds]

    env_states = env_states.detach() + noise_scale * torch.randn_like(
        env_states, device=device
    )
    model_states = model_states.detach() + noise_scale * torch.randn_like(
        model_states, device=device
    )

    env_states.requires_grad = True
    model_states.requires_grad = True
    # Create the optimizer
    opt = torch.optim.Adam([env_states, model_states], lr=learning_rate)
    initial_states = torch.cat((env_states, model_states), dim=1).detach().cpu().numpy()

    # Run the optimization
    iter_count = 1
    q_model_prev = torch.full((n_inits,), float("nan"), device=device)
    q_env_prev = torch.full((n_inits,), float("nan"), device=device)
    while True:
        # Compute q and dq for the current states
        (
            action,
            hidden_step,
            env_states_step,
            joint_states_step,
        ) = model.forward_step_coupled(
            env_states, context_inputs, model_states, joint_states
        )

        q_model = 0.5 * torch.sum(
            (hidden_step.squeeze() - model_states.squeeze()) ** 2, dim=1
        )
        q_env = 0.5 * torch.sum(
            (env_states_step.squeeze() - env_states.squeeze()) ** 2, dim=1
        )

        dq_model = torch.abs(q_model - q_model_prev)
        dq_env = torch.abs(q_env - q_env_prev)

        q_model_scalar = torch.mean(q_model)
        q_env_scalar = torch.mean(q_env)

        q_scalar = q_model_scalar + q_env_scalar
        q = q_model + q_env
        dq = dq_model + dq_env

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress (in-place; overwrites previous line)
        if iter_count % 10 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(
                f"\r  iter {iter_count}/{max_iters}  "
                f"q={mean_q:.2E}+/-{std_q:.2E}  "
                f"dq={mean_dq:.2E}+/-{std_dq:.2E}",
                end="",
                flush=True,
            )

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("\n  Maximum iteration count reached. Terminating.")
            break
        q_model_prev = q_model
        q_env_prev = q_env
        iter_count += 1
    # Collect fixed points
    states = torch.cat((env_states, model_states), dim=1)
    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )

    print(f"\nFound {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:  # TODO: Fix this
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps


def find_fixed_points_dt(
    model: pl.LightningModule,
    state_trajs: np.array,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if len(inputs.shape) > 1:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)

    # Select the initial states
    states = state_pts[idx]
    if len(inputs.shape) > 1:
        inputs = inputs[idx]
    else:
        inputs = inputs.unsqueeze(0).repeat(n_inits, 1)

    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)

    # Require gradients for the states
    states = states.detach()
    initial_states = states.detach().cpu().numpy()
    states.requires_grad = True

    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)

    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    x_store = np.zeros((n_inits, max_iters, state_dim))
    q_store = np.zeros((n_inits, max_iters))
    while True:
        # Compute q and dq for the current states
        x_store[:, iter_count - 1, :] = states.cpu().detach().numpy()
        q_store[:, iter_count - 1] = q_prev.cpu().detach().numpy()
        _, F = model.decoder(inputs, states)
        q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress (in-place; overwrites previous line)
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(
                f"\r  iter {iter_count}/{max_iters}  "
                f"q={mean_q:.2E}+/-{std_q:.2E}  "
                f"dq={mean_dq:.2E}+/-{std_dq:.2E}",
                end="",
                flush=True,
            )

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("\n  Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        q_store[:, iter_count - 1] = q_prev.cpu().detach().numpy()
        iter_count += 1
    # Collect fixed points

    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )

    print(f"\nFound {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            _, F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, inputs, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps


def find_fixed_points_batched(
    model: pl.LightningModule,
    state_trajs,
    inputs,
    n_inits=4096,
    batch_size=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device=None,
    seed=0,
    compute_jacobians=False,
    log_every=500,
):
    """Memory-efficient batched FP search, intended for (but not limited to) GPU.

    The single-batch ``find_fixed_points`` materialises the full ``n_inits``
    set of states + optimizer state on the chosen device, which OOMs once
    ``n_inits`` gets large. This variant splits the inits into chunks of
    ``batch_size``, runs the FP optimization independently on each chunk, and
    concatenates the results. Output is bit-equivalent (up to floating-point
    nondeterminism) to running the same total ``n_inits`` un-chunked.

    Jacobians, when requested, are computed once over all surviving fixed
    points via ``vectorize=True`` instead of looping one-by-one.

    Parameters
    ----------
    device : str | torch.device | None
        Defaults to ``"cuda"`` if available, else ``"cpu"``.
    batch_size : int
        Number of initial states optimized in one forward/backward step. Tune
        downward if you hit OOM; tune upward for throughput on big GPUs.
    log_every : int
        Print q / dq summary every N iterations of every batch. Set to 0 to
        silence per-batch logging.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    if not torch.is_tensor(state_trajs):
        state_trajs = torch.as_tensor(state_trajs)
    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    # Flatten the trajectory bank, draw n_inits random sample points, then
    # gather the matching input vectors. Matches find_fixed_points semantics.
    if state_trajs.ndim > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if inputs.ndim > 1 and inputs.shape[0] == n_samples and inputs.shape[1] == n_steps:
            inputs_flat = inputs.reshape(-1, inputs.shape[-1])
        else:
            inputs_flat = None
    else:
        state_pts = state_trajs
        state_dim = state_pts.shape[-1]
        inputs_flat = inputs if inputs.ndim > 1 else None
    idx_pool = state_pts.shape[0]

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    all_idx = torch.randint(idx_pool, size=(n_inits,), device=device, generator=g)

    n_batches = (n_inits + batch_size - 1) // batch_size

    xstar_chunks = []
    qstar_chunks = []
    dq_chunks = []
    init_chunks = []
    inputs_at_fp_chunks = []
    q_history_per_batch = []

    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, n_inits)
        bsz = hi - lo
        if log_every:
            print(f"[batched FP] batch {b + 1}/{n_batches}  ({bsz} inits, device={device})")

        idx = all_idx[lo:hi]
        states = state_pts[idx].detach().clone()
        if inputs_flat is not None:
            cur_inputs = inputs_flat[idx]
        elif inputs.ndim == 1:
            cur_inputs = inputs.unsqueeze(0).repeat(bsz, 1)
        else:
            cur_inputs = inputs[:bsz]

        if noise_scale > 0:
            states = states + noise_scale * torch.randn_like(states)
        init_chunks.append(states.detach().cpu().numpy())
        states.requires_grad = True
        opt = torch.optim.Adam([states], lr=learning_rate)

        q_history = {"iter": [], "mean": [], "median": [], "p10": [], "p90": []}
        history_every = max(1, min(50, max_iters // 200 if max_iters >= 200 else 1))
        q_prev = torch.full((bsz,), float("nan"), device=device)
        q = None
        dq = None

        for it in range(1, max_iters + 1):
            F = model(cur_inputs, states)
            q = 0.5 * torch.sum((F.squeeze(-2) - states.squeeze(-2)) ** 2, dim=-1)
            (q.mean()).backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                dq = torch.abs(q - q_prev)
                if it == 1 or it % history_every == 0 or it == max_iters:
                    qv = q.detach().cpu().numpy()
                    q_history["iter"].append(it)
                    q_history["mean"].append(float(np.mean(qv)))
                    q_history["median"].append(float(np.median(qv)))
                    q_history["p10"].append(float(np.percentile(qv, 10)))
                    q_history["p90"].append(float(np.percentile(qv, 90)))
                if log_every and it % log_every == 0:
                    qn = q.detach().cpu().numpy()
                    dqn = dq.detach().cpu().numpy()
                    print(
                        f"\r  iter {it}/{max_iters}  "
                        f"q={qn.mean():.2E}+/-{qn.std():.2E}  "
                        f"dq={dqn.mean():.2E}+/-{dqn.std():.2E}",
                        end="",
                        flush=True,
                    )
                q_prev = q.detach()

        xstar_chunks.append(states.detach().cpu().numpy())
        qstar_chunks.append(q.detach().cpu().numpy())
        dq_chunks.append(dq.detach().cpu().numpy())
        inputs_at_fp_chunks.append(cur_inputs.detach().cpu().numpy())
        q_history_per_batch.append(q_history)

        # Free chunk-local tensors before the next batch.
        del states, opt, q, dq, q_prev, F, cur_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    xstar = np.concatenate(xstar_chunks, axis=0).squeeze()
    qstar = np.concatenate(qstar_chunks, axis=0)
    dq_arr = np.concatenate(dq_chunks, axis=0)
    x_init = np.concatenate(init_chunks, axis=0)
    inputs_at_fp = np.concatenate(inputs_at_fp_chunks, axis=0)

    # Merge the per-batch q-histories — iters line up, average summaries.
    merged_history = {"iter": [], "mean": [], "median": [], "p10": [], "p90": []}
    if q_history_per_batch:
        merged_history["iter"] = q_history_per_batch[0]["iter"]
        for stat in ("mean", "median", "p10", "p90"):
            stacked = np.stack([np.asarray(h[stat]) for h in q_history_per_batch])
            merged_history[stat] = stacked.mean(axis=0).tolist()
    merged_history = {k: np.asarray(v, dtype=np.float64) for k, v in merged_history.items()}

    all_fps = FixedPoints(
        xstar=xstar,
        x_init=x_init,
        qstar=qstar,
        dq=dq_arr,
        n_iters=np.full_like(qstar, max_iters),
    )
    all_fps.q_history = merged_history

    print(f"\nFound {len(all_fps.xstar)} candidate fixed points across {n_batches} batches.")

    if compute_jacobians:
        x_t = torch.as_tensor(xstar, device=device, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        inp_t = torch.as_tensor(inputs_at_fp, device=device, dtype=torch.float32)

        def _f(x):
            return model(inp_t, x).squeeze(-2)

        try:
            J = torch.autograd.functional.jacobian(_f, x_t, vectorize=True)
            n_fp = x_t.shape[0]
            J = J[torch.arange(n_fp), :, torch.arange(n_fp), :]
        except (RuntimeError, TypeError) as e:
            print(f"[batched FP] vectorised jacobian failed ({e}); falling back to per-FP loop.")
            J_list = []
            for i in range(x_t.shape[0]):
                inp_i = inp_t[i].unsqueeze(0)
                x_i = x_t[i].unsqueeze(0)
                J_i = torch.autograd.functional.jacobian(
                    lambda x: model(inp_i, x).squeeze(-2), x_i
                )
                J_list.append(J_i.squeeze())
            J = torch.stack(J_list)

        all_fps.J_xstar = J.detach().cpu().numpy()
        all_fps.decompose_jacobians()

    return all_fps
