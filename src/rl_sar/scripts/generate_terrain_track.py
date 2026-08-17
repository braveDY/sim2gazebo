#!/usr/bin/env python3
"""
Generate a continuous 1D obstacle course (Terrain Track) along the X-axis for Gazebo Classic.
Base ground is model://ground_plane at Z = 0.0.
Flat markers have visual-only links to avoid conflicting contacts with ground_plane.
"""

import math
import os
import random


def generate_world():
    random.seed(42)

    links = []

    def add_box(name, x, y, z_top, size_x, size_y, height, color_ambient, color_diffuse):
        z_center = z_top - height / 2.0
        link_str = f"""
            <link name="{name}">
                <pose>{x:.4f} {y:.4f} {z_center:.4f} 0 0 0</pose>
                <collision name="{name}_col">
                    <geometry>
                        <box><size>{size_x:.4f} {size_y:.4f} {height:.4f}</size></box>
                    </geometry>
                </collision>
                <visual name="{name}_vis">
                    <geometry>
                        <box><size>{size_x:.4f} {size_y:.4f} {height:.4f}</size></box>
                    </geometry>
                    <material>
                        <ambient>{color_ambient}</ambient>
                        <diffuse>{color_diffuse}</diffuse>
                    </material>
                </visual>
            </link>"""
        links.append(link_str)

    def add_visual_marker(name, x, y, size_x, size_y, color_ambient, color_diffuse):
        link_str = f"""
            <link name="{name}">
                <pose>{x:.4f} {y:.4f} 0.001 0 0 0</pose>
                <visual name="{name}_vis">
                    <geometry>
                        <box><size>{size_x:.4f} {size_y:.4f} 0.002</size></box>
                    </geometry>
                    <material>
                        <ambient>{color_ambient}</ambient>
                        <diffuse>{color_diffuse}</diffuse>
                    </material>
                </visual>
            </link>"""
        links.append(link_str)

    def add_cylinder(name, x, y, z_top, radius, height, color_ambient, color_diffuse):
        z_center = z_top - height / 2.0
        link_str = f"""
            <link name="{name}">
                <pose>{x:.4f} {y:.4f} {z_center:.4f} 0 0 0</pose>
                <collision name="{name}_col">
                    <geometry>
                        <cylinder><radius>{radius:.4f}</radius><length>{height:.4f}</length></cylinder>
                    </geometry>
                </collision>
                <visual name="{name}_vis">
                    <geometry>
                        <cylinder><radius>{radius:.4f}</radius><length>{height:.4f}</length></cylinder>
                    </geometry>
                    <material>
                        <ambient>{color_ambient}</ambient>
                        <diffuse>{color_diffuse}</diffuse>
                    </material>
                </visual>
            </link>"""
        links.append(link_str)

    def add_ramp(name, x_center, y_center, z_center, size_x, size_y, thickness, pitch_rad, color_ambient, color_diffuse):
        link_str = f"""
            <link name="{name}">
                <pose>{x_center:.4f} {y_center:.4f} {z_center:.4f} 0 {pitch_rad:.4f} 0</pose>
                <collision name="{name}_col">
                    <geometry>
                        <box><size>{size_x:.4f} {size_y:.4f} {thickness:.4f}</size></box>
                    </geometry>
                </collision>
                <visual name="{name}_vis">
                    <geometry>
                        <box><size>{size_x:.4f} {size_y:.4f} {thickness:.4f}</size></box>
                    </geometry>
                    <material>
                        <ambient>{color_ambient}</ambient>
                        <diffuse>{color_diffuse}</diffuse>
                    </material>
                </visual>
            </link>"""
        links.append(link_str)

    LANE_WIDTH = 2.8

    # -------------------------------------------------------------
    # 0. Start Area: X in [-1.0, 2.0] (Visual ground marker only, no collision conflict)
    # -------------------------------------------------------------
    add_visual_marker(
        "zone0_start_mat",
        x=0.5, y=0.0, size_x=3.0, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )

    # -------------------------------------------------------------
    # 1. Discrete Boxes Grid: X in [2.0, 6.5] (10 cols x 6 rows)
    # -------------------------------------------------------------
    box_size = 0.45
    cols, rows = 10, 6
    x_start = 2.0 + box_size / 2.0
    y_start = -(rows * box_size) / 2.0 + box_size / 2.0

    for c in range(cols):
        for r in range(rows):
            bx = x_start + c * box_size
            by = y_start + r * box_size
            h = random.uniform(0.03, 0.10)
            add_box(
                f"zone1_box_{c}_{r}",
                x=bx, y=by, z_top=h,
                size_x=box_size - 0.005, size_y=box_size - 0.005, height=h,
                color_ambient="0.4 0.3 0.2 1", color_diffuse="0.65 0.45 0.25 1"
            )

    # -------------------------------------------------------------
    # Transition 1: X in [6.5, 8.0]
    # -------------------------------------------------------------
    add_visual_marker(
        "trans1_mat",
        x=7.25, y=0.0, size_x=1.5, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )

    # -------------------------------------------------------------
    # 2. Pyramid Stairs: X in [8.0, 12.8] (5 steps up, flat top, 5 steps down)
    # -------------------------------------------------------------
    step_w = 0.30
    step_h = 0.08
    num_steps = 5

    # Stairs Up
    curr_x = 8.0
    for s in range(num_steps):
        s_x = curr_x + step_w / 2.0
        s_h = (s + 1) * step_h
        add_box(
            f"zone2_stair_up_{s}",
            x=s_x, y=0.0, z_top=s_h,
            size_x=step_w, size_y=LANE_WIDTH, height=s_h,
            color_ambient="0.35 0.35 0.35 1", color_diffuse="0.55 0.55 0.55 1"
        )
        curr_x += step_w

    # Top Platform
    top_len = 1.2
    top_h = num_steps * step_h
    add_box(
        "zone2_stairs_top",
        x=curr_x + top_len / 2.0, y=0.0, z_top=top_h,
        size_x=top_len, size_y=LANE_WIDTH, height=top_h,
        color_ambient="0.4 0.4 0.4 1", color_diffuse="0.6 0.6 0.6 1"
    )
    curr_x += top_len

    # Stairs Down
    for s in range(num_steps):
        s_x = curr_x + step_w / 2.0
        s_h = top_h - (s + 1) * step_h
        if s_h < 0.01:
            s_h = 0.01
        add_box(
            f"zone2_stair_down_{s}",
            x=s_x, y=0.0, z_top=s_h,
            size_x=step_w, size_y=LANE_WIDTH, height=s_h,
            color_ambient="0.35 0.35 0.35 1", color_diffuse="0.55 0.55 0.55 1"
        )
        curr_x += step_w

    # -------------------------------------------------------------
    # Transition 2: X in [12.8, 14.0]
    # -------------------------------------------------------------
    add_visual_marker(
        "trans2_mat",
        x=13.4, y=0.0, size_x=1.2, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )

    # -------------------------------------------------------------
    # 3. Gaps / Trenches: X in [14.0, 19.5]
    # -------------------------------------------------------------
    GAP_ELEV = 0.25
    curr_x = 14.0

    ramp_entry_len = 0.6
    add_box(
        "zone3_entry_step",
        x=curr_x + ramp_entry_len / 2.0, y=0.0, z_top=GAP_ELEV,
        size_x=ramp_entry_len, size_y=LANE_WIDTH, height=GAP_ELEV,
        color_ambient="0.45 0.25 0.15 1", color_diffuse="0.7 0.35 0.2 1"
    )
    curr_x += ramp_entry_len

    gaps = [0.18, 0.22, 0.26, 0.30]
    plank_w = 0.55

    for i, gap in enumerate(gaps):
        curr_x += gap
        p_x = curr_x + plank_w / 2.0
        add_box(
            f"zone3_gap_plank_{i}",
            x=p_x, y=0.0, z_top=GAP_ELEV,
            size_x=plank_w, size_y=LANE_WIDTH, height=GAP_ELEV,
            color_ambient="0.45 0.25 0.15 1", color_diffuse="0.7 0.35 0.2 1"
        )
        curr_x += plank_w

    add_box(
        "zone3_exit_step",
        x=curr_x + ramp_entry_len / 2.0, y=0.0, z_top=GAP_ELEV,
        size_x=ramp_entry_len, size_y=LANE_WIDTH, height=GAP_ELEV,
        color_ambient="0.45 0.25 0.15 1", color_diffuse="0.7 0.35 0.2 1"
    )
    curr_x += ramp_entry_len

    # -------------------------------------------------------------
    # Transition 3: X in [curr_x, curr_x + 1.5]
    # -------------------------------------------------------------
    add_visual_marker(
        "trans3_mat",
        x=curr_x + 0.75, y=0.0, size_x=1.5, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )
    curr_x += 1.5

    # -------------------------------------------------------------
    # 4. Slopes: X in [curr_x, curr_x + 5.2]
    # -------------------------------------------------------------
    ramp_len = 2.0
    ramp_rise = 0.45
    pitch = math.atan2(ramp_rise, ramp_len)
    ramp_thick = 0.06

    up_center_x = curr_x + ramp_len / 2.0
    up_center_z = ramp_rise / 2.0
    add_ramp(
        "zone4_ramp_up",
        x_center=up_center_x, y_center=0.0, z_center=up_center_z,
        size_x=math.sqrt(ramp_len**2 + ramp_rise**2), size_y=LANE_WIDTH, thickness=ramp_thick,
        pitch_rad=-pitch,
        color_ambient="0.35 0.35 0.2 1", color_diffuse="0.65 0.6 0.3 1"
    )
    curr_x += ramp_len

    peak_len = 1.2
    add_box(
        "zone4_ramp_peak",
        x=curr_x + peak_len / 2.0, y=0.0, z_top=ramp_rise,
        size_x=peak_len, size_y=LANE_WIDTH, height=ramp_rise,
        color_ambient="0.35 0.35 0.2 1", color_diffuse="0.65 0.6 0.3 1"
    )
    curr_x += peak_len

    down_center_x = curr_x + ramp_len / 2.0
    down_center_z = ramp_rise / 2.0
    add_ramp(
        "zone4_ramp_down",
        x_center=down_center_x, y_center=0.0, z_center=down_center_z,
        size_x=math.sqrt(ramp_len**2 + ramp_rise**2), size_y=LANE_WIDTH, thickness=ramp_thick,
        pitch_rad=pitch,
        color_ambient="0.35 0.35 0.2 1", color_diffuse="0.65 0.6 0.3 1"
    )
    curr_x += ramp_len

    # -------------------------------------------------------------
    # Transition 4: X in [curr_x, curr_x + 1.5]
    # -------------------------------------------------------------
    add_visual_marker(
        "trans4_mat",
        x=curr_x + 0.75, y=0.0, size_x=1.5, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )
    curr_x += 1.5

    # -------------------------------------------------------------
    # 5. Stepping Stones / Stakes: X in [curr_x, curr_x + 4.5]
    # -------------------------------------------------------------
    num_stakes = 8
    stake_radius = 0.16
    stake_spacing = 0.50
    y_offset = 0.28

    for k in range(num_stakes):
        sx_l = curr_x + k * stake_spacing
        sz_l = random.uniform(0.06, 0.12)
        add_cylinder(
            f"zone5_stake_left_{k}",
            x=sx_l, y=y_offset, z_top=sz_l,
            radius=stake_radius, height=sz_l,
            color_ambient="0.2 0.35 0.2 1", color_diffuse="0.3 0.6 0.35 1"
        )
        sx_r = curr_x + (k + 0.5) * stake_spacing
        sz_r = random.uniform(0.06, 0.12)
        add_cylinder(
            f"zone5_stake_right_{k}",
            x=sx_r, y=-y_offset, z_top=sz_r,
            radius=stake_radius, height=sz_r,
            color_ambient="0.2 0.35 0.2 1", color_diffuse="0.3 0.6 0.35 1"
        )

    curr_x += num_stakes * stake_spacing + 0.5

    # -------------------------------------------------------------
    # Transition 5: X in [curr_x, curr_x + 1.5]
    # -------------------------------------------------------------
    add_visual_marker(
        "trans5_mat",
        x=curr_x + 0.75, y=0.0, size_x=1.5, size_y=LANE_WIDTH,
        color_ambient="0.3 0.3 0.35 1", color_diffuse="0.45 0.5 0.6 1"
    )
    curr_x += 1.5

    # -------------------------------------------------------------
    # 6. Narrow Bridge: X in [curr_x, curr_x + 4.0]
    # -------------------------------------------------------------
    bridge_h = 0.15
    ramp_b_len = 0.5
    bridge_len = 3.0

    add_box(
        "zone6_bridge_entry",
        x=curr_x + ramp_b_len / 2.0, y=0.0, z_top=bridge_h,
        size_x=ramp_b_len, size_y=0.6, height=bridge_h,
        color_ambient="0.4 0.2 0.3 1", color_diffuse="0.7 0.3 0.45 1"
    )
    curr_x += ramp_b_len

    add_box(
        "zone6_narrow_bridge",
        x=curr_x + bridge_len / 2.0, y=0.0, z_top=bridge_h,
        size_x=bridge_len, size_y=0.38, height=bridge_h,
        color_ambient="0.4 0.2 0.3 1", color_diffuse="0.7 0.3 0.45 1"
    )
    curr_x += bridge_len

    add_box(
        "zone6_bridge_exit",
        x=curr_x + ramp_b_len / 2.0, y=0.0, z_top=bridge_h,
        size_x=ramp_b_len, size_y=0.6, height=bridge_h,
        color_ambient="0.4 0.2 0.3 1", color_diffuse="0.7 0.3 0.45 1"
    )
    curr_x += ramp_b_len

    # -------------------------------------------------------------
    # 7. Finish Platform: X in [curr_x, curr_x + 3.0]
    # -------------------------------------------------------------
    add_visual_marker(
        "zone7_finish_mat",
        x=curr_x + 1.5, y=0.0, size_x=3.0, size_y=LANE_WIDTH,
        color_ambient="0.2 0.35 0.4 1", color_diffuse="0.3 0.6 0.7 1"
    )

    # Combine into SDF World
    world_content = f"""<?xml version="1.0" ?>
<!-- Auto-generated Continuous Terrain Track along X-axis for Go2 RL deployment -->
<sdf version="1.5">
    <world name="default">
        <physics type="ode">
            <max_step_size>0.0005</max_step_size>
            <real_time_factor>1</real_time_factor>
            <real_time_update_rate>2000</real_time_update_rate>
            <gravity>0 0 -9.81</gravity>
            <ode>
                <solver>
                    <type>quick</type>
                    <iters>50</iters>
                    <sor>1.3</sor>
                </solver>
                <constraints>
                    <cfm>0.0</cfm>
                    <erp>0.2</erp>
                    <contact_max_correcting_vel>10.0</contact_max_correcting_vel>
                    <contact_surface_layer>0.001</contact_surface_layer>
                </constraints>
            </ode>
        </physics>

        <scene>
            <sky>
                <clouds>
                    <speed>12</speed>
                </clouds>
            </sky>
        </scene>

        <!-- Global light source -->
        <include>
            <uri>model://sun</uri>
        </include>

        <!-- Base Ground Plane (Z = 0.0) -->
        <include>
            <uri>model://ground_plane</uri>
        </include>

        <!-- Continuous Terrain Track along X-axis -->
        <model name="terrain_track">
            <static>true</static>
            {''.join(links)}
        </model>

    </world>
</sdf>
"""
    return world_content


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "worlds")
    os.makedirs(output_dir, exist_ok=True)
    world_file = os.path.join(output_dir, "terrain_track.world")
    content = generate_world()
    with open(world_file, "w") as f:
        f.write(content)
    print(f"Successfully generated continuous terrain track world to: {world_file}")
