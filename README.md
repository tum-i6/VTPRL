
![image](/resources/sim_ai4di_image.png?raw=true)

# Main

Virtual Training Platform for Robot Learning

## Getting started

Start the VTPRL simulator located in the simulator folder, depending on your OS. Once the simulator is started, you should see the Unity logo and a grey background with a lower panel including the "quit" button. The simulator starts a gRPC server and waits for requests on port 9092 (configurable in configuration.xml). At this point, the Python client should be connected to instantiate the manipulator environments in the simulator.

## Installation

While the simulator should run out of the box, to run the Python/agent scripts, you'll need to install some dependencies. Recommended installation on local machines (Windows/Linux/Mac) is with Docker, so you need to install the latest Docker for Linux/Mac or Docker-Desktop and WSL (https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers, https://learn.microsoft.com/en-us/windows/wsl/install) for Windows. Install the CUDA (NVIDIA Container Toolkit) if your machine has an NVIDIA GPU. On Linux, you can enable GPU acceleration by installing [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker). GPU acceleration under Windows is possible with a functioning WSL and Docker-Desktop installation - check: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute). Note that DART is not supported natively in Windows, so you will not be able to use it on Windows without Docker.

For the setup with Docker configuration, you can find a Docker image you can start from in the _Docker_ folder of the repo; commands for building the image and creating a container out of it are provided in Docker/Commands.

# Running locally

First, start your simulator, after which it will load and wait for connections. Following the commands explained in Docker/Commands.txt navigate to the root folder of the repo and run agent/main.py. This should start an example code that instantiates one environment, which appears in the simulator, and starts the training. See main.py for other options.

## Configuration

1. To configure the simulator, you need to modify the configuration.xml file. The file is read once the simulator is started. If you modify a setting, you need to restart the simulator. If there is an error in the XML after you modify it, the simulator will use default settings (check Troubleshooting). The following settings are relevant:

- **PortNumber** - The port on which the gRPC server from the simulator waits for requests. If you want to modify this (e.g., starting several simulators on different ports, or the port is busy), make sure you change this also on the Python side (in config.py).

- **EnableObservationImage** - Whether to provide an image observation from the simulator in addition to the numeric observation (joint speeds, angles ...). Only set this to true if you need to use image observations for your task, as it slows down the performance significantly. If this is enabled, it will return as many images as there are _<CameraParameters>_ elements, with the specified resolution in _ObservationImageWidth_ and _ObservationImageHeight_.

- **CameraParameters**: Here, you specify the position and rotation of a camera from which you want to view the environment. Camera position is specified in x, y, z coordinates in meters (note that the y value is height in Unity coordinate frame), and rotation in Euler angles in degrees. You can specify multiple CameraParameters elements to be able to view the environment from different angles (You can use the "Switch view" button in the GUI to change views but note that if you also specify _EnableObservationImage_ to True it will slow down the simulation significantly as it will need to render images for each camera at each timestep).

- **MaxJointVelocity** - When controlling the robot in Joint Velocity control, you should pass actions with normalized values in the range \[-1, 1]. These values are then multiplied by MaxJointVelocity to calculate the speed with which the joint should rotate in the positive or negative direction. There is no clipping done, so make sure you always pass normalized action values in the \[-1, 1] range. The current max joint velocity is set to ~57 deg/sec (1 rad/sec). Please do not modify it if not needed, and talk with us about how to do it if you need to, as it might have some side effects.

2. For configuring the Python agent side, there is a config.py file with a parameter dictionary that you can change; there are inline comments explaining the parameters. Besides, the main.py file includes a method for creating sample vectorized environments and the way to set them for RL policy training or control them by model-based policies. Finally, the iiwa_sample_env.py file is an example gym environment for the reach and balance task, showing how and where the state, reward, and terminal condition should be defined.

# Troubleshooting 

1. If there are errors related to the simulator, you can check the Player.log file that is created when you are running the simulator. The log file is located under an OS-specific folder, which you can find [here](https://docs.unity3d.com/Manual/LogFiles.html) under _Player-related log locations_, with TUM-CIT-AIR and VTPRL-Simulator for _Company name_ and _Product name_ placeholders.

2. Building a Docker image on Windows gets stuck at a specific stage/percent: You should probably increase the resources (RAM, CPUs, Memory) that you allow Docker to use from your host in Docker -> Advanced settings.
 
3. grpc._channel._Rendezvous: Rendezvous of RPC that terminated with: status = StatusCode.UNAVAILABLE - for some reason, the Python client cannot connect to the simulator. Possible reasons:
- The simulator is not running.
- The configuration.xml and config.py have different ports set.
- If you are trying to connect the client from inside the Docker container on Windows, you have not specified the ip_address to 'host.docker.internal' in config.py.
- Maybe the simulator cannot start the gRPC server on the port provided in configuration.xml - check the simulator log file Player.log.

# Authors and acknowledgment
The work has been performed in the project AI4DI: Artificial Intelligence for Digitizing Industry, under grant agreement No. 826060. The project is co-funded by grants from Germany, Austria, Finland, France, Norway, Latvia, Belgium, Italy, Switzerland, and the Czech Republic, and by the Electronic Component Systems for European Leadership Joint Undertaking (ECSEL JU).
