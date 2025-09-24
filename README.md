
![image](/resources/sim_a-iq-ready_image.png?raw=true)

# Main

Virtual Training Platform for Robot Learning

## Getting started

Start the VTPRL simulator located in the simulator folder, depending on your OS. Once the simulator is started, you should see the VTPRL logo and a grey background with a lower panel including the "quit" button. The simulator starts a gRPC server and waits for requests on port 9092 (configurable in configuration.xml). At this point, the Python client should be connected to instantiate the manipulator/warehouse environments in the simulator.

## Installation

While the simulator should run out of the box, to run the Python/agent scripts, you'll need to install some dependencies. Recommended installation on local machines (Windows/Linux/Mac) is with Docker, so you need to install the latest Docker for Linux/Mac or Docker-Desktop and WSL (https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers, https://learn.microsoft.com/en-us/windows/wsl/install) for Windows. Install the CUDA (NVIDIA Container Toolkit) if your machine has an NVIDIA GPU. On Linux, you can enable GPU acceleration by installing [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker). GPU acceleration under Windows is possible with a functioning WSL and Docker-Desktop installation - check: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute). Note that DART is not supported natively in Windows, so you will not be able to use it on Windows without Docker.

For the setup with Docker configuration, you can find a Docker image you can start from in the _Docker_ folder of the repo; commands for building the image and creating a container out of it are provided in Docker/Commands.

# Running locally

First, start your simulator, after which it will load and wait for connections. Following the commands explained in Docker/Commands.txt navigate to the root folder of the repo and run agent/main.py. This should start an example code that instantiates one environment, which appears in the simulator, and starts the model-based evaluation. See main.py for other options.

## Configuration

1. To configure the simulator, you need to modify the configuration.xml file. The file is read once the simulator is started. If you modify a setting, you need to restart the simulator. If there is an error in the XML after you modify it, the simulator will use default settings (check Troubleshooting). Please refer to [Configuration-Parameters](/docs/Configuration-Parameters.md) for a complete description of all parameters settings.

2. For configuring the Python agent side, there is a config.py file with a parameter dictionary that you can change; there are inline comments explaining the parameters. Besides, the main.py file includes a method for creating sample vectorized environments and the way to set them for RL policy training or control them by model-based policies. Finally, the iiwa_sample_env.py file is an example gym environment for the reach and balance task, showing how and where the state, reward, and terminal condition should be defined. In the same manner, warehouse_unity_env.py file is an example gym environment for the mobile robot navigation task.

# Troubleshooting 

1. If there are errors related to the simulator, you can check the Player.log file that is created when you are running the simulator. The log file is located under an OS-specific folder, which you can find [here](https://docs.unity3d.com/Manual/LogFiles.html) under _Player-related log locations_, with TUM-CIT-AIR and VTPRL-Simulator for _Company name_ and _Product name_ placeholders.

2. Building a Docker image on Windows gets stuck at a specific stage/percent: You should probably increase the resources (RAM, CPUs, Memory) that you allow Docker to use from your host in Docker -> Advanced settings.
 
3. grpc._channel._Rendezvous: Rendezvous of RPC that terminated with: status = StatusCode.UNAVAILABLE - for some reason, the Python client cannot connect to the simulator. Possible reasons:
- The simulator is not running.
- The configuration.xml and config.py have different ports set.
- If you are trying to connect the client from inside the Docker container on Windows, you have not specified the ip_address to 'host.docker.internal' in config.py.
- Maybe the simulator cannot start the gRPC server on the port provided in configuration.xml - check the simulator log file Player.log.

# Authors and acknowledgment
The work has been performed in the following two projects:
- AI4DI: Artificial Intelligence for Digitizing Industry, under grant agreement No. 826060. The project is co-funded by grants from Germany, Austria, Finland, France, Norway, Latvia, Belgium, Italy, Switzerland, and the Czech Republic, and by the Electronic Component Systems for European Leadership Joint Undertaking (ECSEL JU).
- A-IQ READY: Artificial Intelligence using Quantum Measured Information for Realtime Distributed Systems at the Edge, under grant agreement No. 101096658. The project is funded within the Chips Joint Undertaking (Chips JU) - the Public-Private Partnership for research, development, and innovation under Horizon Europe – and National Authorities.