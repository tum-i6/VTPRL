import traceback
import xml.etree.ElementTree as ET
import copy

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

    # Port #
    field = root.find('CommunicationType')
    oldFieldText = field.text
    newFieldText = config.config_dict["communication_type"]
    if(newFieldText != "GRPC" and newFieldText != "ROS" and newFieldText != "ZMQ"):
        print("Given CommunicationType field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    # Port #
    field = root.find('PortNumber')
    oldFieldText = field.text
    field.text = config.config_dict["port_number"]
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    # Control cycles #
    field = root.find('TimestepDurationInSeconds')
    oldFieldText = field.text
    field.text = str(config.dart_dict["control_cycle"])
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    field = root.find('PhyscsSimulationIncrementInSeconds')
    oldFieldText = field.text
    field.text = str(config.dart_dict["unity_cycle"])
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    # Images #
    field = root.find('EnableObservationImage')
    oldFieldText = field.text
    newFieldText = str(config.config_dict["use_images"]).lower()
    if(newFieldText != "true" and newFieldText != "false"):
        print("Given EnableObservationImage field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    field = root.find('ObservationImageWidth')
    oldFieldText = field.text
    field.text = str(config.config_dict["image_size"])
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    field = root.find('ObservationImageHeight')
    oldFieldText = field.text
    field.text = str(config.config_dict["image_size"])
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    # Camera position #
    field = root.find('ObservationCameras')[0][0]
    x = field[0]
    oldFieldText = x.text
    x.text = str(config.config_dict["camera_position"][0])
    if(oldFieldText != x.text):
        x.set('updated', 'yes')

    y = field[1]
    oldFieldText = y.text
    y.text = str(config.config_dict["camera_position"][1])
    if(oldFieldText != y.text):
        y.set('updated', 'yes')

    z = field[2]
    oldFieldText = z.text
    z.text = str(config.config_dict["camera_position"][2])
    if(oldFieldText != z.text):
        z.set('updated', 'yes')

    # Camera rotation #
    field = root.find('ObservationCameras')[0][1]
    x = field[0]
    oldFieldText = x.text
    x.text = str(config.config_dict["camera_rotation"][0])
    if(oldFieldText != x.text):
        x.set('updated', 'yes')

    y = field[1]
    oldFieldText = y.text
    y.text = str(config.config_dict["camera_rotation"][1])
    if(oldFieldText != y.text):
        y.set('updated', 'yes')

    z = field[2]
    oldFieldText = z.text
    z.text = str(config.config_dict["camera_rotation"][2])
    if(oldFieldText != z.text):
        z.set('updated', 'yes')

    # End effector #
    field = root.find('EnableEndEffector')
    oldFieldText = field.text
    newFieldText = str(config.config_dict["enable_end_effector"]).lower()
    if(newFieldText != "true" and newFieldText != "false"):
        print("Given EnableEndEffector field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    field = root.find('EndEffectorModel')
    oldFieldText = field.text
    if config.config_dict["robotic_tool"] == "3_gripper":
        model = "ROBOTIQ_3F"
    elif config.config_dict["robotic_tool"] == "2_gripper":
        model = "ROBOTIQ_2F85"
    elif config.config_dict["robotic_tool"] == "calibration_pin":
        model = "CALIBRATION_PIN"
    else:
        model = "ROBOTIQ_3F"

    field.text = model
    if(oldFieldText != field.text):
        field.set('updated', 'yes')

    # Box #
    try: # Might not be availavle depending on the simulator version
        field = root.find('ItemSizeX')
        oldFieldText = field.text
        field.text = str(config.goal_dict["box_dim"][0])
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

        field = root.find('ItemSizeY')
        oldFieldText = field.text
        field.text = str(config.goal_dict["box_dim"][1])
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

        field = root.find('ItemSizeZ')
        oldFieldText = field.text
        field.text = str(config.goal_dict["box_dim"][2])
        if(oldFieldText != field.text):
            field.set('updated', 'yes')
    except:
        pass

    # Randomization #
    field = root.find('RandomizeLatency')
    oldFieldText = field.text
    newFieldText = str(config.randomization_dict["randomize_latency"]).lower()
    if(newFieldText != "true" and newFieldText != "false"):
        print("Given RandomizeLatency field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    field = root.find('RandomizeTorque')
    oldFieldText = field.text
    newFieldText = str(config.randomization_dict["randomize_torque"]).lower()
    if(newFieldText != "true" and newFieldText != "false"):
        print("Given RandomizeTorque field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    field = root.find('RandomizeAppearance')
    oldFieldText = field.text
    newFieldText = str(config.randomization_dict["randomize_appearance"]).lower()
    if(newFieldText != "true" and newFieldText != "false"):
        print("Given RandomizeAppearance field is invalid")
    else:
        field.text = newFieldText
        if(oldFieldText != field.text):
            field.set('updated', 'yes')

    try: # Might not be availavle depending on the simulator version
        field = root.find('EnableShadows')
        oldFieldText = field.text
        newFieldText = str(config.randomization_dict["enable_shadows"]).lower()
        if(newFieldText != "true" and newFieldText != "false"):
            print("Given EnableShadows field is invalid")
        else:
            field.text = newFieldText
            if(oldFieldText != field.text):
                field.set('updated', 'yes')

        field = root.find('ShadowType')
        oldFieldText = field.text
        newFieldText = config.randomization_dict["shadow_type"]
        if(newFieldText != "Soft" and newFieldText != "Hard"):
            print("Given ShadowType field is invalid")
        else:
            field.text = newFieldText
            if(oldFieldText != field.text):
                field.set('updated', 'yes')
    except:
        pass

    try:
        # Write the updated .xml file
        tree.write(xml_file)

    except Exception as e:
        print("Warning: the .xml file has not been updated properly. An Exception was occurred:")
        print(traceback.format_exc())

        oldTree.write(xml_file) # Revert to the old .xml file

        return False

    return True
