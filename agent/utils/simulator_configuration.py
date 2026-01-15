import traceback
import xml.etree.ElementTree as ET
import copy
import math

# ------------------------
# XML helper functions
# ------------------------
def _to_float(val):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip()
            # Reject empty strings
            if s == "":
                return None
            return float(s)
    except Exception:
        return None
    return None

def _texts_equivalent(old_text, new_val):
    # Normalize None and empty-string handling
    if new_val is None:
        # Caller uses None to mean "no change"; treat as equivalent
        return True
    # If old is empty/None and new is empty string (or whitespace), treat as equal
    if (old_text is None or (isinstance(old_text, str) and old_text.strip() == "")) and (isinstance(new_val, str) and new_val.strip() == ""):
        return True
    new_text = str(new_val)
    # Quick path: exact match ignoring surrounding whitespace
    if isinstance(old_text, str) and old_text.strip() == new_text.strip():
        return True
    # Numeric equivalence: treat 1 and 1.0 as equal
    a = _to_float(old_text)
    b = _to_float(new_val)
    if a is not None and b is not None:
        try:
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        except Exception:
            return False
    return False

def set_text(parent, tag, value):
    if parent is None or value is None:
        return
    elem = parent.find(tag)
    if elem is None:
        return
    old = elem.text
    if not _texts_equivalent(old, value):
        elem.text = str(value)
        elem.set('updated', 'yes')

def set_bool(parent, tag, value):
    set_text(parent, tag, str(bool(value)).lower())

def set_vec3(parent, tag, values):
    if parent is None or values is None:
        return
    node = parent.find(tag)
    if node is None or not isinstance(values, (list, tuple)) or len(values) < 3:
        return
    changed = False
    for name, idx in (('x',0), ('y',1), ('z',2)):
        child = node.find(name)
        if child is not None:
            old = child.text
            if not _texts_equivalent(old, values[idx]):
                child.text = str(values[idx])
                changed = True
    if changed:
        node.set('updated', 'yes')

def set_rgba(parent, tag, rgba):
    if parent is None or rgba is None:
        return
    node = parent.find(tag)
    if node is None or not isinstance(rgba, (list, tuple)) or len(rgba) < 4:
        return
    changed = False
    for name, idx in (('r',0), ('g',1), ('b',2), ('a',3)):
        child = node.find(name)
        if child is not None:
            old = child.text
            if not _texts_equivalent(old, rgba[idx]):
                child.text = str(rgba[idx])
                changed = True
    if changed:
        node.set('updated', 'yes')

def set_enum(parent, tag, value, allowed_values):
    if parent is None or value is None:
        return
    if allowed_values and value not in allowed_values:
        print(f"Given {tag} value '{value}' is invalid. Allowed: {allowed_values}")
        return
    set_text(parent, tag, value)

def set_float_array(parent, tag, values):
    if parent is None or not isinstance(values, (list, tuple)):
        return
    node = parent.find(tag)
    if node is None:
        return
    floats = list(node.findall('float'))
    changed = False

    # Update common indices
    min_len = min(len(floats), len(values))
    for i in range(min_len):
        old = floats[i].text
        if not _texts_equivalent(old, values[i]):
            floats[i].text = str(values[i])
            changed = True

    # Add missing elements
    if len(values) > len(floats):
        for i in range(len(floats), len(values)):
            new_elem = ET.Element('float')
            new_elem.text = str(values[i])
            node.append(new_elem)
        changed = True

    # Remove extra elements
    if len(values) < len(floats):
        for i in range(len(values), len(floats)):
            node.remove(floats[i])
        changed = True

    if changed:
        node.set('updated', 'yes')

def get_root_param(config, key):
    """Return a root-level parameter."""
    root = getattr(config, 'root_dict', None)
    if isinstance(root, dict):
        if key in root:
            return root.get(key)
    return None

def get_sim_param(config, key):
    """Return a simulation-level parameter from its proper scope."""
    sim = getattr(config, 'simulation_dict', None)
    if isinstance(sim, dict):
        if key in sim:
            return sim.get(key)
    return None

def get_manip_param(config, key):
    """Return a manipulator-environment parameter from its proper scope."""
    manip = getattr(config, 'manipulator_environment_dict', None)
    if isinstance(manip, dict):
        if key in manip:
            return manip.get(key)
        floor = manip.get('floor') if isinstance(manip.get('floor'), dict) else None
        if floor and key in floor:
            return floor.get(key)
    return None

def get_warehouse_param(config, key):
    """Return a warehouse-environment parameter from its proper scope."""
    ware = getattr(config, 'warehouse_environment_dict', None)
    if isinstance(ware, dict):
        if key in ware:
            return ware.get(key)
        ground = ware.get('ground') if isinstance(ware.get('ground'), dict) else None
        if ground and key in ground:
            return ground.get(key)
    return None

def get_observation_param(config, key):
    """Return an observation parameter from its proper scope."""
    obs = getattr(config, 'observation_dict', None)
    if isinstance(obs, dict):
        if key in obs:
            return obs.get(key)
        laser = obs.get('laser_scan') if isinstance(obs.get('laser_scan'), dict) else None
        if laser and key in laser:
            return laser.get(key)
    return None

def ensure_children(parent, child_tag, desired_count):
    if parent is None:
        return []
    try:
        desired = int(desired_count) if desired_count is not None else 0
    except Exception:
        desired = 0
    existing = list(parent.findall(child_tag))
    if len(existing) == 0:
        if desired > 0:
            print(f"Warning: cannot create '{child_tag}' without a template under '{parent.tag}'.")
        return []
    template = existing[0]
    changed = False
    while len(existing) < desired:
        new_elem = copy.deepcopy(template)
        parent.append(new_elem)
        existing.append(new_elem)
        changed = True
    while len(existing) > desired:
        parent.remove(existing[-1])
        existing.pop()
        changed = True
    if changed:
        parent.set('updated', 'yes')
    return existing

def _clear_updated_flags(root_node):
    if root_node is None:
        return
    for elem in root_node.iter():
        if 'updated' in elem.attrib:
            try:
                del elem.attrib['updated']
            except Exception:
                pass

def update_simulator_configuration(config, xml_file):
    """
        Update the UNITY simulator .xml file based on the settings that the user has been provided in a Config() type of class

        Note: to change more options either adapt the UNITY simulator manually or extend this function, and the Config() class

        :param config: configuration object
        :param xml_file: path of the .xml file

        :return: bool. Returns False when an error was encountered and the .xml file could not be updated properly
    """

    tree = ET.parse(xml_file)
    oldTree = copy.deepcopy(tree)

    root = tree.getroot()

    # Clear previous run 'updated' flags so only current changes are marked
    _clear_updated_flags(root)

    # Simulation section
    simulation = root.find('Simulation')

    # Communication and networking
    set_enum(simulation, 'CommunicationType', get_sim_param(config, 'communication_type'), {'GRPC','GRPC_NRP','ROS','ZMQ'})
    ip_address = get_sim_param(config, 'ip_address')
    if ip_address in ['localhost', 'host.docker.internal']:
        ip_address = '127.0.0.1'
    set_text(simulation, 'IPAddress', ip_address)
    set_text(simulation, 'PortNumber', get_sim_param(config, 'port_number'))

    # Time steps
    set_text(simulation, 'TimestepDurationInSeconds', get_sim_param(config, 'timestep_duration_in_seconds'))
    set_text(simulation, 'PhysicsSimulationIncrementInSeconds', get_sim_param(config, 'physics_simulation_increment_in_seconds'))

    # Simulation flags and seed
    set_bool(simulation, 'ImprovedPatchFriction', get_sim_param(config, 'improved_patch_friction'))
    set_text(simulation, 'RandomSeed', get_sim_param(config, 'random_seed'))
    set_bool(simulation, 'Evaluation', get_sim_param(config, 'evaluation'))

    # Physics randomization
    set_bool(simulation, 'RandomizeEnvironmentPhysics', get_sim_param(config, 'randomize_environment_physics'))
    set_bool(simulation, 'RandomizeTorque', get_sim_param(config, 'randomize_torque'))

    # Persist/replay episode manifests
    set_bool(simulation, 'PersistEpisodeManifests', get_sim_param(config, 'persist_episode_manifests'))
    set_bool(simulation, 'ReplayEpisodeManifest', get_sim_param(config, 'replay_episode_manifest'))
    set_text(simulation, 'ReplayManifestPath', get_sim_param(config, 'replay_manifest_path'))

    # Environment Mode
    set_enum(root, 'EnvironmentMode', get_root_param(config, 'environment_mode'), {'Manipulator','Warehouse'})

    # Manipulator Environment #
    manipulator_env = root.find('ManipulatorEnvironment')

    # Manipulator Model
    set_enum(manipulator_env, 'ManipulatorModel', get_manip_param(config, 'manipulator_model'), {'IIWA14','SO100'})

    # End effector
    set_bool(manipulator_env, 'EnableEndEffector', get_manip_param(config, 'enable_end_effector'))
    set_enum(manipulator_env, 'EndEffectorModel', get_manip_param(config, 'end_effector_model'), {'ROBOTIQ_3F','ROBOTIQ_2F85','CALIBRATION_PIN'})

    # Floor settings #
    floor = manipulator_env.find('Floor') if manipulator_env is not None else None
    # Floor type/material and visualization
    ft_val = get_manip_param(config, 'floor_type')
    set_enum(floor, 'FloorType', str(ft_val).upper() if ft_val is not None else None, {'CHECKERBOARD','WOOD','MONOCHROMATIC'})
    set_vec3(floor, 'FloorSize', get_manip_param(config, 'floor_size'))
    fm_val = get_manip_param(config, 'floor_material')
    set_enum(floor, 'FloorMaterial', str(fm_val).upper() if fm_val is not None else None, {'HOMOGENEOUS','HETEROGENEOUS'})
    set_bool(floor, 'VisualizeFloorMaterial', get_manip_param(config, 'visualize_floor_material'))
    set_rgba(floor, 'FloorMaterialColor', get_manip_param(config, 'floor_material_color'))
    set_float_array(floor, 'FloorMaterialGridX', get_manip_param(config, 'floor_material_grid_x'))
    set_float_array(floor, 'FloorMaterialGridY', get_manip_param(config, 'floor_material_grid_y'))
    set_float_array(floor, 'FloorMaterialGridZ', get_manip_param(config, 'floor_material_grid_z'))
    set_float_array(floor, 'FloorMaterialGridDynamicFriction', get_manip_param(config, 'floor_material_grid_dynamic_friction'))
    set_float_array(floor, 'FloorMaterialGridStaticFriction', get_manip_param(config, 'floor_material_grid_static_friction'))

    # Items section (support arrays)
    items = manipulator_env.find('Items') if manipulator_env is not None else None
    items_cfg_list = get_manip_param(config, 'items')
    if isinstance(items_cfg_list, list) and len(items_cfg_list) > 0:
        item_nodes = ensure_children(items, 'ItemParameters', len(items_cfg_list))
        for i, item_node in enumerate(item_nodes):
            ic = items_cfg_list[i] if i < len(items_cfg_list) else {}
            it_val = ic.get('item_type')
            set_enum(item_node, 'ItemType', str(it_val).upper() if it_val is not None else None, {'SPHERE','BOX'})
            set_vec3(item_node, 'ItemSize', ic.get('item_size'))
            set_text(item_node, 'ItemMass', ic.get('item_mass'))
            set_vec3(item_node, 'ItemCenterOfMass', ic.get('item_center_of_mass'))
            set_text(item_node, 'ItemLinearDamping', ic.get('item_linear_damping'))
            set_bool(item_node, 'ItemObservability', ic.get('item_observability'))
            im_val = ic.get('item_material')
            set_enum(item_node, 'ItemMaterial', str(im_val).upper() if im_val is not None else None, {'HOMOGENEOUS','HETEROGENEOUS'})
            set_bool(item_node, 'VisualizeItemMaterial', ic.get('visualize_item_material'))
            set_rgba(item_node, 'ItemMaterialColor', ic.get('item_material_color'))
            set_rgba(item_node, 'TargetMaterialColor', ic.get('target_material_color'))
            set_float_array(item_node, 'ItemMaterialGridX', ic.get('item_material_grid_x'))
            set_float_array(item_node, 'ItemMaterialGridY', ic.get('item_material_grid_y'))
            set_float_array(item_node, 'ItemMaterialGridZ', ic.get('item_material_grid_z'))
            set_float_array(item_node, 'ItemMaterialGridDynamicFriction', ic.get('item_material_grid_dynamic_friction'))
            set_float_array(item_node, 'ItemMaterialGridStaticFriction', ic.get('item_material_grid_static_friction'))
            set_bool(item_node, 'RandomizeItemMass', ic.get('randomize_item_mass'))
            set_text(item_node, 'ItemMassRandomizationRange', ic.get('item_mass_randomization_range'))
            set_bool(item_node, 'RandomizeItemCenterOfMass', ic.get('randomize_item_center_of_mass'))
            set_vec3(item_node, 'ItemCenterOfMassRandomizationRange', ic.get('item_center_of_mass_randomization_range'))
            set_bool(item_node, 'RandomizeItemFriction', ic.get('randomize_item_friction'))
            set_text(item_node, 'ItemDynamicFrictionRandomizationRange', ic.get('item_dynamic_friction_randomization_range'))
            set_text(item_node, 'ItemStaticFrictionRandomizationRange', ic.get('item_static_friction_randomization_range'))

    # Manipulator TrajectoryString (if provided)
    set_text(manipulator_env, 'TrajectoryString', get_manip_param(config, 'trajectory_string'))

    # Warehouse Environment section
    warehouse_env = root.find('WarehouseEnvironment')
    # Top-level
    set_enum(warehouse_env, 'AMRModel', get_warehouse_param(config, 'amr_model'), {'SAFELOG_S2'})
    set_bool(warehouse_env, 'EnableTransport', get_warehouse_param(config, 'enable_transport'))
    set_text(warehouse_env, 'MaxChassisLinearSpeed', get_warehouse_param(config, 'max_chassis_linear_speed'))
    set_text(warehouse_env, 'MaxChassisAngularSpeed', get_warehouse_param(config, 'max_chassis_angular_speed'))
    set_text(warehouse_env, 'WheelDriveForceLimit', get_warehouse_param(config, 'wheel_drive_force_limit'))
    set_text(warehouse_env, 'WheelDriveDamping', get_warehouse_param(config, 'wheel_drive_damping'))

    # Ground
    ground = warehouse_env.find('Ground') if warehouse_env is not None else None
    set_enum(ground, 'GroundType', get_warehouse_param(config, 'ground_type'), {'MONOCHROMATIC','TEXTURED','PREFAB'})
    set_vec3(ground, 'GroundSize', get_warehouse_param(config, 'ground_size'))
    set_text(ground, 'WallHeight', get_warehouse_param(config, 'wall_height'))
    set_enum(ground, 'GroundMaterial', get_warehouse_param(config, 'ground_material'), {'HOMOGENEOUS','HETEROGENEOUS'})
    set_bool(ground, 'VisualizeGroundMaterial', get_warehouse_param(config, 'visualize_ground_material'))
    set_rgba(ground, 'GroundMaterialColor', get_warehouse_param(config, 'ground_material_color'))
    set_float_array(ground, 'GroundMaterialGridX', get_warehouse_param(config, 'ground_material_grid_x'))
    set_float_array(ground, 'GroundMaterialGridY', get_warehouse_param(config, 'ground_material_grid_y'))
    set_float_array(ground, 'GroundMaterialGridZ', get_warehouse_param(config, 'ground_material_grid_z'))
    set_float_array(ground, 'GroundMaterialGridDynamicFriction', get_warehouse_param(config, 'ground_material_grid_dynamic_friction'))
    set_float_array(ground, 'GroundMaterialGridStaticFriction', get_warehouse_param(config, 'ground_material_grid_static_friction'))

    # Items (support arrays)
    warehouse_items = warehouse_env.find('Items') if warehouse_env is not None else None
    warehouse_items_cfg = get_warehouse_param(config, 'items')
    if isinstance(warehouse_items_cfg, list) and len(warehouse_items_cfg) > 0:
        wi_nodes = ensure_children(warehouse_items, 'ItemParameters', len(warehouse_items_cfg))
        for i, item_node in enumerate(wi_nodes):
            ic = warehouse_items_cfg[i] if i < len(warehouse_items_cfg) else {}
            it_val = ic.get('item_type')
            set_enum(item_node, 'ItemType', str(it_val).upper() if it_val is not None else None, {'SPHERE','BOX'})
            set_vec3(item_node, 'ItemSize', ic.get('item_size'))
            set_text(item_node, 'ItemMass', ic.get('item_mass'))
            set_vec3(item_node, 'ItemCenterOfMass', ic.get('item_center_of_mass'))
            set_text(item_node, 'ItemLinearDamping', ic.get('item_linear_damping'))
            set_bool(item_node, 'ItemObservability', ic.get('item_observability'))
            im_val = ic.get('item_material')
            set_enum(item_node, 'ItemMaterial', str(im_val).upper() if im_val is not None else None, {'HOMOGENEOUS','HETEROGENEOUS'})
            set_bool(item_node, 'VisualizeItemMaterial', ic.get('visualize_item_material'))
            set_rgba(item_node, 'ItemMaterialColor', ic.get('item_material_color'))
            set_rgba(item_node, 'TargetMaterialColor', ic.get('target_material_color'))
            set_float_array(item_node, 'ItemMaterialGridX', ic.get('item_material_grid_x'))
            set_float_array(item_node, 'ItemMaterialGridY', ic.get('item_material_grid_y'))
            set_float_array(item_node, 'ItemMaterialGridZ', ic.get('item_material_grid_z'))
            set_float_array(item_node, 'ItemMaterialGridDynamicFriction', ic.get('item_material_grid_dynamic_friction'))
            set_float_array(item_node, 'ItemMaterialGridStaticFriction', ic.get('item_material_grid_static_friction'))
            set_bool(item_node, 'RandomizeItemMass', ic.get('randomize_item_mass'))
            set_text(item_node, 'ItemMassRandomizationRange', ic.get('item_mass_randomization_range'))
            set_bool(item_node, 'RandomizeItemCenterOfMass', ic.get('randomize_item_center_of_mass'))
            set_vec3(item_node, 'ItemCenterOfMassRandomizationRange', ic.get('item_center_of_mass_randomization_range'))
            set_bool(item_node, 'RandomizeItemFriction', ic.get('randomize_item_friction'))
            set_text(item_node, 'ItemDynamicFrictionRandomizationRange', ic.get('item_dynamic_friction_randomization_range'))
            set_text(item_node, 'ItemStaticFrictionRandomizationRange', ic.get('item_static_friction_randomization_range'))

    # Obstacle manager
    set_bool(warehouse_env, 'EnableObstacleManager', get_warehouse_param(config, 'enable_obstacle_manager'))
    set_text(warehouse_env, 'ObstaclePlacementSeparationMultiplier', get_warehouse_param(config, 'obstacle_placement_separation_multiplier'))
    set_text(warehouse_env, 'ObstacleSpawnBoundaryMargin', get_warehouse_param(config, 'obstacle_spawn_boundary_margin'))

    # Static obstacles (support arrays)
    set_text(warehouse_env, 'StaticObstacleCount', get_warehouse_param(config, 'static_obstacle_count'))
    statics = warehouse_env.find('StaticObstacles') if warehouse_env is not None else None
    static_list = get_warehouse_param(config, 'static_obstacles')
    if isinstance(static_list, list) and len(static_list) > 0:
        sop_nodes = ensure_children(statics, 'StaticObstacleParameters', len(static_list))
        for i, sop in enumerate(sop_nodes):
            so = static_list[i] if i < len(static_list) else {}
            set_enum(sop, 'ObstacleType', so.get('obstacle_type'), {'BOX'})
            set_vec3(sop, 'ObstacleSize', so.get('obstacle_size'))
            set_bool(sop, 'ObstacleObservability', so.get('obstacle_observability', False))
            set_rgba(sop, 'ObstacleMaterialColor', so.get('obstacle_material_color'))

    # Dynamic obstacles
    dyn = warehouse_env.find('DynamicObstacles') if warehouse_env is not None else None
    dd = get_warehouse_param(config, 'dynamic_obstacles') or {}
    set_text(dyn, 'DynamicObstacleCount', dd.get('dynamic_obstacle_count'))
    set_bool(dyn, 'DynamicObstacleObservability', dd.get('dynamic_obstacle_observability'))
    set_enum(dyn, 'DynamicObstacleMotion', dd.get('dynamic_obstacle_motion'), {'None','Random','Circle','Linear'})
    set_text(dyn, 'DynamicObstacleMinDistanceFromRobot', dd.get('dynamic_obstacle_min_distance_from_robot'))
    set_text(dyn, 'MaxLinearSpeed', dd.get('max_linear_speed'))
    set_text(dyn, 'MaxAngularSpeed', dd.get('max_angular_speed'))
    set_text(dyn, 'RandomMotionChangePeriodSeconds', dd.get('random_motion_change_period_seconds'))
    set_text(dyn, 'LinearMotionCycleSeconds', dd.get('linear_motion_cycle_seconds'))
    set_text(dyn, 'LinearMotionTravelSpeed', dd.get('linear_motion_travel_speed'))
    set_text(dyn, 'CircleMotionTravelSpeed', dd.get('circle_motion_travel_speed'))
    set_text(dyn, 'CircleMotionCurvature', dd.get('circle_motion_curvature'))

    # Observation section #
    observation = root.find('Observation')
    
    # Image settings
    set_bool(observation, 'EnableObservationImage', get_observation_param(config, 'enable_observation_image'))
    set_bool(observation, 'SaveObservationImageAsFile', get_observation_param(config, 'save_observation_image_as_file'))
    set_enum(observation, 'ObservationImageEncoding', get_observation_param(config, 'observation_image_encoding'), {'JPG','PNG'})
    set_text(observation, 'ObservationImageQuality', get_observation_param(config, 'observation_image_quality'))
    set_text(observation, 'ObservationImageWidth', get_observation_param(config, 'observation_image_width'))
    set_text(observation, 'ObservationImageHeight', get_observation_param(config, 'observation_image_height'))

    # Image background color
    set_rgba(observation, 'ObservationImageBackgroundColor', get_observation_param(config, 'observation_image_background_color'))

    # Enable segmentation
    set_bool(observation, 'EnableSegmentation', get_observation_param(config, 'enable_segmentation'))

    # Robot segmentation color
    set_rgba(observation, 'RobotSegmentationColor', get_observation_param(config, 'robot_segmentation_color'))

    set_bool(observation, 'EnableShadows', get_observation_param(config, 'enable_shadows'))
    set_enum(observation, 'ShadowType', get_observation_param(config, 'shadow_type'), {'Soft','Hard','None'})

    # Observation cameras (support arrays)
    cameras = observation.find('ObservationCameras') if observation is not None else None
    cam_cfg_list = get_observation_param(config, 'observation_cameras')
    if isinstance(cam_cfg_list, list) and len(cam_cfg_list) > 0:
        cam_nodes = ensure_children(cameras, 'CameraParameters', len(cam_cfg_list))
        for i, cam_node in enumerate(cam_nodes):
            cam_cfg = cam_cfg_list[i] if i < len(cam_cfg_list) else {}
            set_float_array(cam_node, 'CameraPosition', cam_cfg.get('camera_position'))
            set_float_array(cam_node, 'CameraRotation', cam_cfg.get('camera_rotation'))
            set_text(cam_node, 'CameraVerticalFOV', cam_cfg.get('camera_vertical_fov'))

    # Randomization settings
    set_bool(observation, 'RandomizeAppearance', get_observation_param(config, 'randomize_appearance'))
    set_text(observation, 'CameraPositionRandomizationRangeInMeters', get_observation_param(config, 'camera_position_randomization_range_in_meters'))
    set_text(observation, 'CameraRotationRandomizationRangeInDegrees', get_observation_param(config, 'camera_rotation_randomization_range_in_degrees'))

    # Laser scan settings (Warehouse)
    set_bool(observation, 'EnableLaserScan', get_observation_param(config, 'enable_laser_scan'))
    laser = observation.find('LaserScan') if observation is not None else None
    set_text(laser, 'NumMeasurementsPerScan', get_observation_param(config, 'num_measurements_per_scan'))
    set_text(laser, 'RangeMetersMin', get_observation_param(config, 'range_meters_min'))
    set_text(laser, 'RangeMetersMax', get_observation_param(config, 'range_meters_max'))
    set_text(laser, 'ScanAngleStartDegrees', get_observation_param(config, 'scan_angle_start_degrees'))
    set_text(laser, 'ScanAngleEndDegrees', get_observation_param(config, 'scan_angle_end_degrees'))

    try:
        # Write the updated .xml file
        tree.write(xml_file)

    except Exception as e:
        print("Warning: the .xml file has not been updated properly. An Exception was occurred:")
        print(traceback.format_exc())

        oldTree.write(xml_file) # Revert to the old .xml file

        return False

    return True
