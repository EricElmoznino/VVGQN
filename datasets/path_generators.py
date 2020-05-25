import numpy as np
from numpy.random import normal, rayleigh, beta


def noisy_path(environment,
               n_samples=20, m_scale=0.5, r_scale=5,
               min_d_wall=2, r_from_walls=np.pi / 1.5):
    assert environment.agent.radius is not None
    assert environment.agent.radius < min_d_wall
    assert len(environment.rooms) > 0

    p, d = start_state(environment, min_d_wall + 1)
    positions = [p]
    directions = [d]
    movements = [0]
    rotations = [0]

    for _ in range(1, n_samples):
        r = (beta(r_scale, r_scale) - 0.5) * 2 * np.pi
        m = rayleigh(m_scale)

        d_prop = dir_angle(d + r)
        p_prop = p + m * np.array([np.cos(d_prop), -np.sin(d_prop)]) # -sin because of lefthand OpenGL coordinates
        d_wall, r_wall = closest_wall(environment, p_prop, d_prop)
        if d_wall < min_d_wall:
            m = 0
            sign = np.sign(r_wall) if r_wall != 0 else 1
            r = sign * (r_from_walls - np.abs(r_wall))
            d = dir_angle(d + r)
        else:
            d = d_prop
            p = p_prop

        positions.append(p)
        directions.append(d)
        movements.append(m)
        rotations.append(r)

    return {
        'positions': np.stack(positions).astype(np.float32),
        'directions': np.array(directions, dtype=np.float32),
        'movements': np.array(movements, dtype=np.float32),
        'rotations': np.array(rotations, dtype=np.float32)
    }


def rat_path(environment,
             t=15, delta_t=0.1, d_from_walls=2, v_scale=1.4, r_mu=0, r_std=5.7,
             perim_v_reduc=0.25, perim_r_change=np.pi/2):
    assert environment.agent.radius is not None
    assert environment.agent.radius < d_from_walls
    assert len(environment.rooms) > 0

    n_samples = int(t // delta_t)
    v_scale *= delta_t
    r_mu *= delta_t
    r_std *= delta_t

    # Path data to return
    positions = np.zeros((n_samples, 2), dtype=np.float32)
    directions = np.zeros((n_samples), dtype=np.float32)
    movements = np.zeros((n_samples), dtype=np.float32)
    rotations = np.zeros((n_samples), dtype=np.float32)

    # Initialization
    positions[0], directions[0] = start_state(environment, d_from_walls)
    random_rotations = normal(r_mu, r_std, size=n_samples)
    random_movements = rayleigh(v_scale, size=n_samples)
    movement = random_movements[0]

    for step in range(1, n_samples):
        closest_wall_d, closest_wall_r = closest_wall(environment, positions[step-1], directions[step-1])
        if closest_wall_d < d_from_walls and np.abs(closest_wall_r) < perim_r_change:
            sign = np.sign(closest_wall_r) if closest_wall_r != 0 else 1
            rotation = sign * (perim_r_change - np.abs(closest_wall_r)) + random_rotations[step]
            movement = (1 - perim_v_reduc) * movement
        else:
            rotation = random_rotations[step]
            movement = random_movements[step]

        direction = dir_angle(directions[step - 1] + rotation)
        velocity = movement * np.array([np.cos(direction), np.sin(direction)])
        new_pos = positions[step-1] + velocity

        positions[step] = new_pos
        directions[step] = direction
        movements[step] = movement
        rotations[step] = rotation

    return {
        'positions': positions,
        'directions': directions,
        'movements': movements,
        'rotations': rotations
    }


def closest_wall(environment, pos, dir):
    pos = np.array([pos[0], 0, pos[1]])
    current_room = None
    for room in environment.rooms:
        if room.point_inside(pos):
            current_room = room
            break
    if current_room is None:
        return -1, 0

    wall_start_to_pos = pos - current_room.outline
    d_wall_to_pos = np.sum(current_room.edge_norms * wall_start_to_pos, axis=1)
    closest = np.argmin(d_wall_to_pos)

    d_closest = d_wall_to_pos[closest]

    norm_closest = current_room.edge_norms[closest]
    norm_dir = np.arctan2(-norm_closest[2], -norm_closest[0])
    r_closest = dir_angle(dir - norm_dir)

    return d_closest, r_closest


def start_state(environment, d_from_walls):
    # Keep retrying until we find a suitable position
    while True:
        # Pick a room, sample rooms proportionally to floor surface area
        r = environment.rand.choice(environment.rooms, probs=environment.room_probs)

        # Choose a random point within the square bounding box of the room
        pos = environment.rand.float(
            low=[r.min_x + d_from_walls, 0, r.min_z + d_from_walls],
            high=[r.max_x - d_from_walls, 0, r.max_z - d_from_walls]
        )

        # Make sure the position is within the room's outline
        if not r.point_inside(pos):
            continue

        # Make sure the position doesn't intersect with any walls
        if environment.intersect(environment, pos, environment.agent.radius):
            continue

        # Pick a direction
        dir = environment.rand.float(-np.pi, np.pi)

        # Remove y-axis of position
        pos = pos[[0, 2]]

        return pos, dir


def dir_angle(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle
