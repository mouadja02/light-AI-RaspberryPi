FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

ENV LC_ALL=C

ENV LIMIT=""
#"-o Acquire::http::Dl-Limit=400 -o Acquire::https::Dl-Limit=400"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set up the user
ENV UNAME docker
RUN useradd -m $UNAME || true && echo "$UNAME:$UNAME" | chpasswd && adduser $UNAME sudo
RUN mkdir -p /etc/sudoers.d && \
    echo "$UNAME ALL=(ALL) ALL" > /etc/sudoers.d/$UNAME && \
    chmod 0440 /etc/sudoers.d/$UNAME && \
    chown ${UID}:${GID} -R /home/$UNAME

WORKDIR /home/${UNAME}/

#Install the packages
RUN apt-get $LIMIT update -yq \
&& apt-get $LIMIT install -y locales locales-all -yq \
&& apt-get $LIMIT install ssh -yq \
&& apt-get $LIMIT install x11-apps -yq \
&& apt-get $LIMIT install sudo -yq \
&& apt-get $LIMIT install fdisk -yq \
&& apt-get $LIMIT install gparted -yq \
&& apt-get $LIMIT install net-tools -yq \
&& apt-get $LIMIT install iputils-ping -yq \
&& apt-get $LIMIT install iptables -yq \
&& apt-get $LIMIT install nano -yq \
&& apt-get $LIMIT install ubuntu-restricted-addons -yq \
&& apt-get $LIMIT install ubuntu-restricted-extras -yq \
&& apt-get $LIMIT install firefox -yq \
&& apt-get $LIMIT install evince -yq \
&& apt-get $LIMIT install default-jdk -yq \
&& apt-get $LIMIT install git -yq \
&& apt-get $LIMIT install python3-pip -yq \
&& apt-get $LIMIT install cmake libncurses5-dev libncursesw5-dev -yq \
&& apt $LIMIT install libcanberra-gtk-module libcanberra-gtk3-module -yq \
&& apt $LIMIT install wimtools -yq \
&& apt $LIMIT install software-properties-common -yq \
&& apt-get $LIMIT update -yq \
&& apt-get $LIMIT dist-upgrade -yq \
&& apt-get clean -yq \
&& rm -rf /var/lib/apt/lists/* \
&& python3 -m pip install --upgrade pip\
&& python3 -m pip install Pillow\
&& python3 -m pip install scipy\
&& python3 -m pip install jupyter\
&& python3 -m pip install ipywidgets\
&& python3 -m pip install numpy \
&& python3 -m pip install matplotlib\
&& python3 -m pip install pandas

RUN pip3 install --timeout=3000 pybind11 numpy Cython h5py tensorflow tensorflow-datasets torch torchvision torchaudio

ADD ./atom ./atom
ADD ./vscode ./vscode
ADD ./atom/.bashrc ./.bashrc

#Install Atom
RUN apt-get $LIMIT update -yq \
&& dpkg -i ./atom/atom-amd64.deb \
|| true \
&& apt --fix-broken install -yq \
&& dpkg -i ./atom/atom-amd64.deb \
&& apt-get clean -yq \
&& rm -rf /var/lib/apt/lists/*

#Install VSCode
RUN apt-get $LIMIT update -yq \
&& dpkg -i ./vscode/code-amd64.deb \
|| true \
&& apt --fix-broken install -yq \
&& dpkg -i ./vscode/code-amd64.deb \
&& apt-get clean -yq \
&& rm -rf /var/lib/apt/lists/*

