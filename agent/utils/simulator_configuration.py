import traceback
import xml.etree.ElementTree as ET
import copy
import math

# ------------------------
# XML helper functions
# ------------------------
def _to_float(val):
    """Convert a value to float when possible.

    Args:
        val: Candidate numeric/string value.

    Returns:
        Float value when conversion succeeds; otherwise ``None``.
    """
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
    """Compare XML text and candidate value with numeric-aware equivalence.

    Args:
        old_text: Existing XML text content.
        new_val: Candidate replacement value.

    Returns:
        True when both values are considered equivalent, otherwise False.
    """
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
    """Set child text value when it differs from current XML content.

    Args:
        parent: Parent XML element.
        tag: Child tag name to update.
        value: New value to write.
    """
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
    """Set a boolean child value as lowercase XML text.

    Args:
        parent: Parent XML element.
        tag: Child tag name to update.
        value: Boolean-like input.
    """
    set_text(parent, tag, str(bool(value)).lower())

def set_vec3(parent, tag, values):
    """Update a nested ``x/y/z`` XML vector node.

    Args:
        parent: Parent XML element.
        tag: Child vector tag containing ``x``, ``y``, ``z`` nodes.
        values: Sequence with at least three values.
    """
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
    """Update a nested ``r/g/b/a`` XML color node.

    Args:
        parent: Parent XML element.
        tag: Child color tag containing ``r``, ``g``, ``b``, ``a`` nodes.
        rgba: Sequence with at least four values.
    """
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
    """Set an XML enum-like field when value is valid.

    Args:
        parent: Parent XML element.
        tag: Child tag name to update.
        value: Candidate enum value.
        allowed_values: Collection of allowed string values.
    """
    if parent is None or value is None:
        return
    if allowed_values and value not in allowed_values:
        print(f"Given {tag} value '{value}' is invalid. Allowed: {allowed_values}")
        return
    set_text(parent, tag, value)

def set_float_array(parent, tag, values):
    """Synchronize a ``<float>`` array node with a Python sequence.

    Args:
        parent: Parent XML element.
        tag: Child array tag containing ``<float>`` nodes.
        values: Sequence of float-compatible values.
    """
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
    """Return a root-level parameter.

    Args:
        config: Configuration object containing ``root_dict``.
        key: Key to retrieve.

    Returns:
        Parameter value when available; otherwise ``None``.
    """
    root = getattr(config, 'root_dict', None)
    if isinstance(root, dict):
        if key in root:
            return root.get(key)
    return None

def get_sim_param(config, key):
    """Return a simulation-level parameter from its proper scope.

    Args:
        config: Configuration object containing ``simulation_dict``.
        key: Key to retrieve.

    Returns:
        Parameter value when available; otherwise ``None``.
    """
    sim = getattr(config, 'simulation_dict', None)
    if isinstance(sim, dict):
        if key in sim:
            return sim.get(key)
    return None

def get_manip_param(config, key):
    """Return a manipulator-environment parameter from its proper scope.

    Args:
        config: Configuration object containing ``manipulator_environment_dict``.
        key: Key to retrieve.

    Returns:
        Parameter value when available in manipulator scope; otherwise ``None``.
    """
    manip = getattr(config, 'manipulator_environment_dict', None)
    if isinstance(manip, dict):
        if key in manip:
            return manip.get(key)
        floor = manip.get('floor') if isinstance(manip.get('floor'), dict) else None
        if floor and key in floor:
            return floor.get(key)
    return None

def get_warehouse_param(config, key):
    """Return a warehouse-environment parameter from its proper scope.

    Args:
        config: Configuration object containing ``warehouse_environment_dict``.
        key: Key to retrieve.

    Returns:
        Parameter value when available in warehouse scope; otherwise ``None``.
    """
    ware = getattr(config, 'warehouse_environment_dict', None)
    if isinstance(ware, dict):
        if key in ware:
            return ware.get(key)
        ground = ware.get('ground') if isinstance(ware.get('ground'), dict) else None
        if ground and key in ground:
            return ground.get(key)
    return None

def get_observation_param(config, key):
    """Return an observation parameter from its proper scope.

    Args:
        config: Configuration object containing ``observation_dict``.
        key: Key to retrieve.

    Returns:
        Parameter value when available in observation scope; otherwise ``None``.
    """
    obs = getattr(config, 'observation_dict', None)
    if isinstance(obs, dict):
        if key in obs:
            return obs.get(key)
        laser = obs.get('laser_scan') if isinstance(obs.get('laser_scan'), dict) else None
        if laser and key in laser:
            return laser.get(key)
    return None

def ensure_children(parent, child_tag, desired_count):
    """Ensure an XML parent contains exactly the requested number of children.

    Args:
        parent: Parent XML element.
        child_tag: Child element tag to count/synchronize.
        desired_count: Target number of children.

    Returns:
        List of synchronized child elements.
    """
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
    """Remove transient ``updated`` markers from an XML tree.

    Args:
        root_node: Root XML element.
    """
    if root_node is None:
        return
    for elem in root_node.iter():
        if 'updated' in elem.attrib:
            try:
                del elem.attrib['updated']
            except Exception:
                pass

def update_simulator_configuration(config, xml_file):
    """Update Unity's ``configuration.xml`` from a Python ``Config`` object.

    Note:
        To support additional XML fields, extend this function and the
        corresponding ``Config`` dictionary sections.

    Args:
        config: Configuration object containing section dictionaries consumed by
            this XML writer.
        xml_file: Filesystem path of the target XML file.

    Returns:
        ``True`` when the XML file was updated successfully, ``False`` if an
        exception occurred and the previous XML content was restored.
    """

    tree = ET.parse(xml_file)
    oldTree = copy.deepcopy(tree)

    root = tree.getroot()

    # Clear previous run 'updated' flags so only current changes are marked
    _clear_updated_flags(root)

    # Simulation section
    simulation = root.find('Simulation')

    # Communication and networking
    set_enum(simulation, 'CommunicationType', get_sim_param(config, 'communication_type'), {'GRPC','GRPC_NRP','GRPC_SHM','GRPC_BIN','ROS','ZMQ'})
    ip_address = get_sim_param(config, 'ip_address')
    if ip_address in ['localhost', 'host.docker.internal']:
        ip_address = '127.0.0.1'
    set_text(simulation, 'IPAddress', ip_address)
    set_text(simulation, 'PortNumber', get_sim_param(config, 'port_number'))

    # Shared-memory IPC parameters (GRPC_SHM mode)
    set_text(simulation, 'SharedMemoryDirectory', get_sim_param(config, 'shared_memory_directory'))
    set_text(simulation, 'SharedMemoryCapacity', get_sim_param(config, 'shared_memory_capacity'))
    set_text(simulation, 'SharedMemorySlotSizeMB', get_sim_param(config, 'shared_memory_slot_size_mb'))
    set_bool(simulation, 'SharedMemorySameKernel', get_sim_param(config, 'shared_memory_same_kernel'))

    # Profiling
    set_bool(simulation, 'EnableProfiling', get_sim_param(config, 'enable_profiling'))

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
    # Manipulator group (segmented via <Manipulators><ManipulatorParameters><Count>)
    manip_cfg = getattr(config, 'manipulator_environment_dict', None)
    manip_list = manip_cfg.get('manipulators') if isinstance(manip_cfg, dict) else None
    if not isinstance(manip_list, list):
        manip_list = None

    manip_count_int = len(manip_list) if isinstance(manip_list, list) and len(manip_list) > 0 else 1

    manip_parent = manipulator_env.find('Manipulators') if manipulator_env is not None else None
    manip_nodes = ensure_children(manip_parent, 'ManipulatorParameters', manip_count_int)

    for i, manip_node in enumerate(manip_nodes):
        cfg = manip_list[i] if manip_list and i < len(manip_list) else {}
        set_text(manip_node, 'Count', cfg.get('count'))
        set_enum(manip_node, 'ManipulatorModel', cfg.get('manipulator_model'), {'IIWA14','SO100'})
        set_bool(manip_node, 'EnableEndEffector', cfg.get('enable_end_effector'))
        set_enum(manip_node, 'EndEffectorModel', cfg.get('end_effector_model'), {'ROBOTIQ_3F','ROBOTIQ_2F85','CALIBRATION_PIN','DEFAULT_GRIPPER'})
        set_text(manip_node, 'JointDriveStiffness', cfg.get('joint_drive_stiffness'))
        set_text(manip_node, 'JointDriveDamping', cfg.get('joint_drive_damping'))
        set_text(manip_node, 'TrajectoryString', cfg.get('trajectory_string'))
        set_vec3(manip_node, 'TargetSize', cfg.get('target_size'))
        set_rgba(manip_node, 'TargetMaterialColor', cfg.get('target_material_color'))

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
            set_text(item_node, 'Count', ic.get('count'))
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

    # Warehouse Environment section
    warehouse_env = root.find('WarehouseEnvironment')

    # AMR group (segmented via <AMRs><AMRParameters><Count>)
    warehouse_cfg = getattr(config, 'warehouse_environment_dict', None)
    amr_cfg_list = warehouse_cfg.get('amrs') if isinstance(warehouse_cfg, dict) else None
    if not isinstance(amr_cfg_list, list):
        amr_cfg_list = None

    amr_count_int = len(amr_cfg_list) if isinstance(amr_cfg_list, list) and len(amr_cfg_list) > 0 else 1

    amrs_parent = warehouse_env.find('AMRs') if warehouse_env is not None else None
    amr_nodes = ensure_children(amrs_parent, 'AMRParameters', amr_count_int)

    for i, amr_node in enumerate(amr_nodes):
        cfg = amr_cfg_list[i] if amr_cfg_list and i < len(amr_cfg_list) else {}
        set_text(amr_node, 'Count', cfg.get('count'))
        set_enum(amr_node, 'AMRModel', cfg.get('amr_model'), {'SAFELOG_S2'})
        set_bool(amr_node, 'EnableTransport', cfg.get('enable_transport'))
        set_rgba(amr_node, 'RobotSegmentationColor', cfg.get('robot_segmentation_color'))
        set_bool(amr_node, 'RandomizeRobotAppearance', cfg.get('randomize_robot_appearance'))
        set_bool(amr_node, 'EnableLaserScan', cfg.get('enable_laser_scan'))
        laser_cfg = cfg.get('laser_scan') if isinstance(cfg.get('laser_scan'), dict) else {}
        laser_node = amr_node.find('LaserScan') if amr_node is not None else None
        set_text(laser_node, 'RangeMetersMin', laser_cfg.get('range_meters_min'))
        set_text(laser_node, 'RangeMetersMax', laser_cfg.get('range_meters_max'))
        set_text(laser_node, 'ScanAngleStartDegrees', laser_cfg.get('scan_angle_start_degrees'))
        set_text(laser_node, 'ScanAngleEndDegrees', laser_cfg.get('scan_angle_end_degrees'))
        set_text(laser_node, 'NumMeasurementsPerScan', laser_cfg.get('num_measurements_per_scan'))
        set_text(laser_node, 'SensorOffsetX', laser_cfg.get('sensor_offset_x'))
        set_text(laser_node, 'SensorOffsetY', laser_cfg.get('sensor_offset_y'))
        set_text(amr_node, 'MaxChassisLinearSpeed', cfg.get('max_chassis_linear_speed'))
        set_text(amr_node, 'MaxChassisAngularSpeed', cfg.get('max_chassis_angular_speed'))
        set_text(amr_node, 'WheelDriveForceLimit', cfg.get('wheel_drive_force_limit'))
        set_text(amr_node, 'WheelDriveDamping', cfg.get('wheel_drive_damping'))
        set_vec3(amr_node, 'TargetSize', cfg.get('target_size'))
        set_rgba(amr_node, 'TargetMaterialColor', cfg.get('target_material_color'))

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
            set_text(item_node, 'Count', ic.get('count'))
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
    set_bool(warehouse_env, 'EnableNavMesh', get_warehouse_param(config, 'enable_navmesh'))
    set_text(warehouse_env, 'ObstaclePlacementSeparationMultiplier', get_warehouse_param(config, 'obstacle_placement_separation_multiplier'))
    set_text(warehouse_env, 'ObstacleSpawnBoundaryMargin', get_warehouse_param(config, 'obstacle_spawn_boundary_margin'))

    # Static obstacles (support arrays)
    statics = warehouse_env.find('StaticObstacles') if warehouse_env is not None else None
    static_list = get_warehouse_param(config, 'static_obstacles')
    if isinstance(static_list, list) and len(static_list) > 0:
        sop_nodes = ensure_children(statics, 'StaticObstacleParameters', len(static_list))
        for i, sop in enumerate(sop_nodes):
            so = static_list[i] if i < len(static_list) else {}
            set_text(sop, 'Count', so.get('count'))
            set_enum(sop, 'ObstacleType', so.get('obstacle_type'), {'BOX'})
            set_vec3(sop, 'ObstacleSize', so.get('obstacle_size'))
            set_bool(sop, 'ObstacleObservability', so.get('obstacle_observability', False))
            set_text(sop, 'ObstacleMinDistanceFromRobot', so.get('obstacle_min_distance_from_robot'))
            set_rgba(sop, 'ObstacleMaterialColor', so.get('obstacle_material_color'))

    # Dynamic obstacle segments
    dyn_parent = warehouse_env.find('DynamicObstacles') if warehouse_env is not None else None
    dynamic_list = get_warehouse_param(config, 'dynamic_obstacles')
    if isinstance(dynamic_list, list) and len(dynamic_list) > 0:
        dyn_nodes = ensure_children(dyn_parent, 'DynamicObstacleParameters', len(dynamic_list))
        for i, dyn_node in enumerate(dyn_nodes):
            dd = dynamic_list[i] if i < len(dynamic_list) else {}
            set_text(dyn_node, 'Count', dd.get('count'))
            set_enum(dyn_node, 'ObstacleModel', dd.get('obstacle_model'), {'SAFELOG_S2'})
            set_bool(dyn_node, 'ObstacleObservability', dd.get('obstacle_observability'))
            set_enum(dyn_node, 'ObstacleMotion', dd.get('obstacle_motion'), {'None','Random','Circle','Linear'})
            set_text(dyn_node, 'ObstacleMinDistanceFromRobot', dd.get('obstacle_min_distance_from_robot'))
            set_text(dyn_node, 'ObstacleMaxLinearSpeed', dd.get('obstacle_max_linear_speed'))
            set_text(dyn_node, 'ObstacleMaxAngularSpeed', dd.get('obstacle_max_angular_speed'))
            set_text(dyn_node, 'RandomMotionChangePeriodSeconds', dd.get('random_motion_change_period_seconds'))
            set_text(dyn_node, 'LinearMotionCycleSeconds', dd.get('linear_motion_cycle_seconds'))
            set_text(dyn_node, 'LinearMotionTravelSpeed', dd.get('linear_motion_travel_speed'))
            set_text(dyn_node, 'CircleMotionTravelSpeed', dd.get('circle_motion_travel_speed'))
            set_text(dyn_node, 'CircleMotionCurvature', dd.get('circle_motion_curvature'))

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
    set_bool(observation, 'RandomizeRobotAppearance', get_observation_param(config, 'randomize_robot_appearance'))
    set_text(observation, 'CameraPositionRandomizationRangeInMeters', get_observation_param(config, 'camera_position_randomization_range_in_meters'))
    set_text(observation, 'CameraRotationRandomizationRangeInDegrees', get_observation_param(config, 'camera_rotation_randomization_range_in_degrees'))

    try:
        # Write the updated .xml file
        tree.write(xml_file)

    except Exception as e:
        print("Warning: the .xml file has not been updated properly. An Exception was occurred:")
        print(traceback.format_exc())

        oldTree.write(xml_file) # Revert to the old .xml file

        return False

    return True
