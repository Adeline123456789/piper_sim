"""Offline MuJoCo sysid fit to the real05 width and uncalibrated SDK effort proxy (not measured torque)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
from mujoco import sysid
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import configure_rigid_collection, read_recorded_commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    calibration = Path(__file__).resolve().parents[2]
    capture = calibration / 'sysid_housing_shape_20261110/mapped_capture/commanded_probe.csv'
    effort_path = calibration / 'sysid_contact_resolution_20261111/gripper_effort.csv'
    initial_path = calibration / 'sysid_multicontact_20261111/convex_layers_step1/initial_state.npz'
    task_path = Path(__file__).resolve().parents[1] / 'configs/tasks/real_connector_sysid_candidate.yaml'
    task = yaml.safe_load(task_path.read_text())
    sim = MujocoBackend(dict(load(), task_scene=task))
    configure_rigid_collection(sim)
    mujoco.mj_saveLastXML(str(args.output / 'source.xml'), sim.model)
    spec = mujoco.MjSpec.from_file(str(args.output / 'source.xml'))
    spec.add_sensor(name='opening', type=mujoco.mjtSensor.mjSENS_TENDONPOS,
                    objtype=mujoco.mjtObj.mjOBJ_TENDON, objname='right/gripper_width')
    spec.add_sensor(name='effort', type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                    objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname='right/gripper')
    model = spec.compile()
    initial = np.load(initial_path)
    state = sysid.create_initial_state(model, initial['qpos'], initial['qvel'])
    rows, times, actions = read_recorded_commands(capture)
    # Official rollout consumes one control row per physics step: preserve real ZOH sends.
    grid = np.arange(0., 3., model.opt.timestep)
    indices = np.maximum(0, np.searchsorted(times, grid, side='right') - 1)
    controls = np.tile(initial['ctrl'], (len(grid), 1))
    controls[:, sim.indices['right']['ctrl']] = actions[indices]
    control = sysid.TimeSeries.from_control_names(grid, controls, model)
    effort_rows = [r for r in csv.DictReader(effort_path.open()) if r['kind'] == 'feedback']
    et = np.array([float(r['time_s']) for r in effort_rows])
    ev = np.array([float(r['effort_Nm']) for r in effort_rows])
    keep = (times > 0.) & (times < 3.)
    mt = times[keep]
    width = np.array([float(r['feedback_gripper_open']) * .095 for r in rows])[keep]
    measured = sysid.TimeSeries.from_names(mt, np.c_[width, np.interp(mt, et, ev)], model,
                                           names=['opening', 'effort'])
    sequence = sysid.ModelSequences('piper_gripper', spec, 'real05_unloaded', state, control, measured)
    params = sysid.ParameterDict()
    params.add(sysid.Parameter('pd', [1000., 25.], [250., 1.], [20000., 2000.],
                              modifier=lambda s, p: sysid.apply_pdgain(s, 'right/gripper', p.value)))

    def force_scale(s, p):
        s.actuator('right/gripper').forcerange = [-.3 * p.value[0], .3 * p.value[0]]

    def friction(s, p):
        for name in ('right/joint7', 'right/joint8'):
            s.joint(name).frictionloss = p.value[0]

    params.add(sysid.Parameter('force_per_effort', 10. / .3, 1., 100., modifier=force_scale))
    params.add(sysid.Parameter('finger_friction_n', .5, 0., 10., modifier=friction))
    params.add(sysid.Parameter('effort_bias', 0., -.15, .15))

    def sensor_residual(p, predicted, observed, model, return_pred_all, **_):
        predicted = predicted.resample(observed.times)
        values = predicted.data.copy()
        values[:, 1] = values[:, 1] / p['force_per_effort'].value[0] + p['effort_bias'].value[0]
        predicted = sysid.TimeSeries(observed.times, values, observed.signal_mapping)
        residual = (values - observed.data) / [.0002, .02]
        # Fit the moving opening segment; its later settling remains held out.
        return residual[observed.times < 1.2].ravel(), predicted, observed

    residual_fn = sysid.build_residual_fn(models_sequences=[sequence], modify_residual=sensor_residual,
                                         n_threads=4)
    before = residual_fn(params.as_vector(), params, return_pred_all=True)
    fitted, result = sysid.optimize(params, residual_fn, max_iters=6, verbose=True)
    after = residual_fn(fitted.as_vector(), fitted, return_pred_all=True)
    sysid.save_results(args.output, [sequence], params, fitted, result, residual_fn)
    scores = {}
    for label, output in [('before', before), ('after', after)]:
        prediction, truth = output[1][0][0], output[2][0][0]
        error = prediction.data - truth.data
        scores[label] = {}
        for split, mask in [('fit', truth.times < 1.2), ('held_out', truth.times >= 1.2)]:
            rmse = np.sqrt(np.mean(error[mask] ** 2, axis=0))
            scores[label][split] = dict(width_rmse_hardware_mm=float(rmse[0] * 1000 / 1.0173063365777468),
                                        effort_proxy_rmse=float(rmse[1]))
        np.savez(args.output / f'{label}_sensors.npz', times=truth.times,
                 predicted=prediction.data, measured=truth.data)
    assert np.isfinite(result.x).all() and np.isfinite(after[0]).all()
    summary = dict(engine=mujoco.__version__, method='mujoco.sysid.optimize: bounded nonlinear least squares',
                   parameters=fitted.as_vector().tolist(), jacobian_rank=int(np.linalg.matrix_rank(result.jac)),
                   parameter_count=len(result.x), scores=scores, capture=str(capture), effort=str(effort_path),
                   training_eligible=False, hardware_commands=0,
                   limitations='Uncalibrated SDK effort proxy, NOT measured torque. Unloaded identification only. Force/effort conversion, finger inertia and friction are not independently measured. No loaded drop or transfer claim.')
    summary['parameter_names'] = ['kp', 'kv', 'force_per_effort', 'finger_friction_n', 'effort_bias']
    summary['max_iterations'] = 6
    summary['objective_history'] = result.extras['objective']
    sources = [Path(__file__), task_path, capture, effort_path, initial_path, args.output / 'source.xml']
    manifest = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output / 'inputs.sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'identify_plug_gripper.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    sim.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
