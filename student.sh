#! /bin/bash

# Script pour les étudiants ENSEIRB, plusieurs choses à savoir pour les différents systèmes d'exploitation. 

# Pour Linux, tout est fonctionnel, tel quel!

# Pour Mac OS, installer docker Desktop puis décommenter les lignes : 
# xhost + $(hostname)
# xhost - $(hostname)
# Et changer le DISPLAY en ajoutant host.docker.internal:0
# Tout cela permet d'avoir sur l'ecran de l'host les GUI qui s'affichent.

# Pour Windows, il faut commencer par installer le WSL 2. 
# Pour cela, suivez le tuto : https://korben.info/installer-wsl2-windows-linux.html
# Une fois le WSL 2 installé avec Linux et mis par défaut comme dans le tuto, il vous faut mettre à jour WSL 2 avec la commande : 
# wsl --update 
# Cela permettra au DISPLAY d'exporter les GUI correctement. 
# Ensuite, pour faire fonctionner l'application chat, il est nécessaire d'avoir une ip du même réseau que l'hote. Or, de base le WSL 2 est en mode NAT, il faut alors passer en mode bridge. 
# Pour mettre le WSL en mode bridge, il vous faut premièrement vérifier que la virtualisation dans le bios est activé. 
# Ensuite suivre le tuto suivant : https://gkaelin.com/bridger-le-reseau-de-wsl2-dans-windows-11/
# Et si dans ce tuto, l'HyperV n'est pas activé, installer le driverW11, HyperV.bat dans ce même dossier en mode administrateur avant que continuer le tuto. 
   
currentDirectory="$(pwd)"
imageName=student

echo "Searching for Docker image ..."
DOCKER_IMAGE_ID=$(docker images -q $imageName | head -n 1)
echo "Found and using ${DOCKER_IMAGE_ID}"

#xhost + $(hostname)
#host.docker.internal:0
container_id=$(docker run -d --network host --user docker:docker \
 -e DISPLAY=unix$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v `pwd`/Work:/home/docker/Work \
 ${DOCKER_IMAGE_ID} sleep infinity) 

#host.docker.internal:0
docker exec -it --user docker:docker \
 -e DISPLAY=unix$DISPLAY \
 -e TERM=xterm-256color \
 $container_id /bin/bash

docker commit $container_id student
docker stop $container_id
docker rm $container_id

#xhost - $(hostname)
