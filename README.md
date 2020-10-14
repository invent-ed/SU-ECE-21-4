# SU-ECE-21-4

This repository holds the final deliverables of Seattle University (SU) senior design team ECE 21.4 for Panthera Corporation. This software is an on-going collaboration between SU and Panthera, and this repository is a continuation of the work done in the past 4 years by SU ECE (Electrical and Computer Engineering) senior design teams. The past teams were ECE 17.7, ECE 18.7, ECE 19.7, and ECE 20.4.

This repository contains the software Recognition, which uses computer vision to determine the number of unique snow leopards in an image dataset.

## Setup

1. Add photos of snow leopards (JPG) to "data/images"
2. Add templates (BMP with 1-bit color depth) to "data/templates"
    - Regions that contain a snow leopard should be white, and the rest black
3. Edit "data/config.json" to desired configurations

## How to run

1. Open PowerShell/Terminal to project root directory (where this README.md file is located)
2. Build Docker image: `$ docker build . -t panthera`
    - This creates a Docker image called "panthera"
    - This may take a while since it needs to install a lot of Python libraries
3. Start new Docker container with image, and mount "data" folder
    - Windows PowerShell: `$ docker run -it --rm -v $PWD\data:/app/data panthera`
    - Mac/Linux Terminal: `$ docker run -it --rm -v $(pwd)/data:/app/data panthera`
    - The option `-it` makes the container interactive (show program in terminal as it runs)
    - The option `--rm` automatically deletes the container when it is done running
    - The option `-v <host dir>:<container dir>` mounts the "data" folder from the host into the container so that results can be accessed once the program is done running
    
## Authors

- Ross Pitman (Panthera Corporation)
- ECE 19.7 (Seattle University)
- ECE 20.4 (Seattle University)
- ECE 21.4 (Seattle University)