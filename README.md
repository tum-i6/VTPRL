	
 <div align="center">
![image](/resources/sim_ai4di_image.png?raw=true)
</div>

# Main

Virtual Training Platform for Robot Learning

## Getting started

Start the simulator located in the simulator folder depending on your OS. Once the simulator is started, you should see the Unity logo and a sky background with "switch view" and "quit" buttons. The simulator starts a gRPC server and waits for requests on port 9092 (configurable in configuration.xml). At this point, the python client should be connected to instantiate the manipulator environments in the simulator.

## Installation

While the simulator should run out of the box, for running the python/agent scripts you'll need to install some dependencies. Recomended installation on local machines (Windows/Linux) is with Docker, so you need to install the latest Docker for Windows/Linux. On Linux, you can also enable GPU acceleration through Docker if you have NVIDIA GPU, you need to install [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) for this. On Windows there is no GPU support with Docker, if you want GPU support on your local machine you have to install the requirements for running the python code natively (you can use package managers like [Conda](https://docs.conda.io/en/latest/) for Windows and install the requirements specified in Docker/requirements.txt. Note that DART is not supported in Windows natively so you will not be able to use it on Windows without Docker. 

For the Docker installation, you can find a Docker image for Linux/Windows you can start from in the _Docker_ folder of the repo, commands for building the image and creating a container out of it are provided in Docker/Commands.

Linux settings tested: Ubuntu 18.04 with Nvidia Docker installed and Nvidia cuda-enabled GPU

Windows settings tested: Windows 10 Pro, Docker Desktop Comunity (v2.1.0.5, Engine 19.03.5) with Linux Containers enabled

# Running locally

First start your simulator, after which it will load and wait for connections. Following the commands explained in Docker/Commands.txt navigate to the root folder of the repo and run agent/main.py. This should start an example code that instaniates one environment which appears in the simulator and starts the training. See main.py for other options.

## Configuration

1. For configuring the simulatior you need to modify the configuration.xml file. The file is read once the simulator is started, if you modify a setting you need to restart the simulator. If there is an error in the xml after you modify it, the simulator will use default settings (check Troubleshooting). The following settings are relevant:

- **PortNumber** - the port on which the gRPC server from the simulator waits for requests. If you want to modify this (e.g., starting several simulatiors on different ports, or the port is busy), make sure you change this also on the python side (in config.py)

- **EnableObservationImage** - Whether to provide an image observation from the simulator in addition to the numeric observation (joint speeds, angles ..). Only set this to true if you need to use image observations for your task, as it slows down the performance significantly. If this is enabled, it will return as many images as there are _<CameraParameters>_ elements, with the specified resolution in _ObservationImageWidth_ and  _ObservationImageHeight_.

- **CameraParameters**: Here you specify the position and rotation to a camera you want to view the environment from. Camera position is specified in x, y, z coordinates in meters (note that the y value is height in Unity coordinate frame), rotation in Euler engles in degrees. You can specify multiple CameraParameters elements to be able to view the environment from different angles (You can use the "Switch view" button in the GUI to changes views, but note that if you also specify _EnableObservationImage_ to True it will slow down the simulation signifficantly as it will need to render images for each camera at each timestep). 

- **MaxJointVelocity** - when controlling the robot in Joint Velocity control, you should pass actions with normalized values in the range \[-1, 1]. These values are then multiplied by MaxJointVelocity to calculate the speed with which the joint should rotate in the positive or negative direction. There is no clipping done, so make sure you always pass normalized action values in the \[-1, 1] range. The current max joint velocity is set to 57 deg/sec (1 rad/sec), please do not modify it if not needed and talk with us how to do it if you need to as it might have some side-effects. 

2. For configuring the python agent side, there is a config.py file with parameter dictionary that you can change, there are inline comments explaining the parameters. Besides, main.py file includes a method for creating sample vectorized environments and the way to set them for RL policy training or control them by model-based policies. Finally, iiwa_sample_env.py file is an example gym environment for the reach and balance task showing how and where the state, reward and terminal condition should be defined.

# Troubleshooting 

1. If there are errors related to the simulator, you can check the Player.log file that is created when you are running the simulator. The log file is located under OS-specific folder which you can find [here](https://docs.unity3d.com/Manual/LogFiles.html) under _Player-related log locations_, with TUM-I6-AI4DI and ManipulatorEnvironment for _Company name_ and _Product name_ placeholders.

2. Building docker image on Windows gets stuck at specific stage/percent: You should probably increase the resources (RAM, CPUs, Memory) that you allow Docker to use from your host in Docker -> Advanced settings
 
3. grpc._channel._Rendezvous: Rendezvous of RPC that terminated with: status = StatusCode.UNAVAILABLE - for some reason the python client cannot connect to the simulator. Possible reasons:
- The simulator is not running
- The configuration.xml and config.py have different ports set
- if you are trying to connect the client from inside docker container on Windows, you have not specified the ip_address to 'host.docker.internal' in config.py
- maybe the simulator cannot start the gRPC server on the port provided in configuration.xml - check the simulator log file Player.log

# Authors and acknowledgment
The work has been performed in the project AI4DI: Artificial Intelligence for Digitizing Industry, under grant agreement No 826060. The project is co-funded by grants from Germany, Austria, Finland, France, Norway, Latvia, Belgium, Italy, Switzerland, and Czech Republic and - Electronic Component Systems for European Leadership Joint Undertaking (ECSEL JU).
