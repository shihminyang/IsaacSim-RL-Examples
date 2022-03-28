# IsaacSim-RL-Examples

## To-Do
- [ ]  Action space:
	- [ ] Joint velocities
	- [ ] Joint torques
- [ ] Model:
	- [ ] Dueling DQN (Customer model)

## Approach target
> Control robot to approach the target position

- **Target**: target position
- **Observation** (*continuous space*):
	approaching vector and joint positions
- **Action** (*continuous space*):
	Joint positions, velocities, or torques (6 values)
- **Model**:
	SAC (stable baselines 3)
- **Train**:
`./python.sh IsaacSim-RL-Examples/train.py --exp approach_positions -H`
