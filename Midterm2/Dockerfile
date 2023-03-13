# start with a base image
FROM continuumio/miniconda3

# install Microsoft Visual C++ Build Tools for Visual Studio 2017
RUN conda install -y -c anaconda msvc-runtime

# install All Necessary Python Packages
RUN conda create -y -n gym python=3 pip

# activate the conda environment
SHELL ["conda", "run", "-n", "gym", "/bin/bash", "-c"]

# minimum install of OpenAI Gym
RUN pip install gym

# Install pystan; necessary to run the ToyText environments
RUN conda install -y -c conda-forge pystan

# install Atari environments
RUN conda install -y -c anaconda git
RUN pip install git+https://github.com/Kojoley/atari-py.git

# install Box2D environments
RUN conda install -y -c anaconda swig
RUN pip install Box2D

# installs the remaining OpenAI Gym environments
RUN pip install gym[all]

# the last two lines are necessary to avoid some bugs that can occur with Pyglet and the Box2D environments.
RUN pip install pyglet==1.2.4
RUN pip install gym[box2d]

# install Xming
RUN apt-get update && apt-get install -y x11-apps

# start Xming running
ENV DISPLAY=:0

# set the working directory
WORKDIR /app

# start the command prompt
CMD ["/bin/bash"]
