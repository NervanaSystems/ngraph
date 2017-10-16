#!  /bin/bash

if [ -z "$RUN_UID" ] ; then

    echo 'ERROR: Environment variable RUN_UID was not set when run-as-user.sh was run'
    echo '       Running as default user (root, in docker)'
    echo ' '

    exit 1

else

    # The username used in the docker container to map the caller UID to
    #
    # Note 'dockuser' is used in other scripts, notably Makefile.  If you
    # choose to change it here, then you need to change it in all other
    # scripts, or else the builds will break.
    #
    DOCK_USER='dockuser'

    # We will be su'ing using a non-login shell or command, and preserving
    # the environment.  This is done so that env. variables passed in with
    # "docker run --env ..." are honored.
    # Therefore, we need to reset at least HOME=/root ...
    #
    # Note also that /home/dockuser is used in other scripts, notably
    # Makefile.  If you choose to change it here, then you need to change it
    # in all other scripts, or else the builds will break.
    #
    export HOME="/home/${DOCK_USER}"

    # Make sure the home directory is owned by the new user
    if [ -d "${HOME}" ] ; then
      chown "${RUN_UID}" "${HOME}"
    fi

    # Add a user with UID of person running docker (in ${RUN_UID})
    # If $HOME does not yet exist, then it will be created
    adduser --disabled-password --gecos 'Docker-User' -u "${RUN_UID}" "${DOCK_USER}"
    # Add dockuser to the sudo group
    adduser "${DOCK_USER}" sudo

    # If root access is needed in the docker image while running as a normal
    # user, uncomment this and add 'sudo' as a package installed in Dockerfile
    # echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

    if [ -z "$RUN_CMD" ] ; then  # Launch a shell as dockuser
      su -m "${DOCK_USER}" -c /bin/bash
    else                         # Run command as dockuser
      su -m "${DOCK_USER}" -c "${RUN_CMD}"
    fi

fi
