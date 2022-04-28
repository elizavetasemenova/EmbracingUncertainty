# EmbracingUncertainty
Material for Applied Machine Learning Days (AMLD) 2020 workshop "Bayesian Inference: embracing uncertainty": 

https://appliedmldays.org/events/amld-epfl-2020/workshops/hands-on-bayesian-machine-learning-embracing-uncertainty

## Preparation

R+Stan and Python+PyMC3 versions of the code will be provided in this repository. The main tutorial, however, will take place in Julia+Turing. For Julia, use the Docker image as explained below. 

### Before the workshop
Warning !! Make sure that you have at least 30GB of free space !! If you don't have enough memory, please pair up with someone who does (otherwise there might be undesirable consequences for your computer)

1. Install Docker from https://docs.docker.com/install/
* Desktop version for Mac and Windows (Requires creating docker hub account)
* Server version for Linux: Follow instructions under *Install Docker Engine - Community*

2. Verify installation by running the following in a terminal (Mac, Linux) or PowerShell (Windows):

```docker run hello-world```

This should output the following:

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```

3. Adjust used resources in Docker to fit your computer, e.g.
* Mac: Docker -> Preferences -> Disk: set Disk image size to 16GB; Apply
* Windows: Docker -> Settings -> Advanced: set Disk image size to 16GB, CPUs to 1, Memory to 1280MB; Apply

4. Download Docker image from DockerHub.

```docker pull semenovae/julia-workshop```

### During the workshop

1. Run the Julia environment

``` docker run -p 8888:8888 semenovae/julia-workshop  ```

2. Create a new Jupyter notebook

3. At the end of the workshop, make sure to download your Jupyter notebook before ending the Docker session and deleting the Docker image
```
Ctrl+C

docker ps // To obtain container ID

docker rm container-id -f

```

### Misc (FYI)
1. Create docker image from Dockerfile and push it to Docker Hub

```
docker build -t your_dockerID/your_image_name:1 .
docker tag your_dockerID/your_image_name:1 dockerID/your_image_name:latest
docker push your_dockerID/your_image_name:1
docker push your_dockerID/your_image_name:latest
```

2. List Docker images

```
docker images
```

3. Remove all docker images

```
docker system prune -a
```




