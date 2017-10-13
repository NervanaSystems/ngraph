#!  /bin/bash

if [ -z "$RUN_UID" ] ; then

    echo 'ERROR: Environment variable RUN_UID was not set when run-as-user.sh was run'
    echo '       Running as default user (root, in docker)'
    echo ' '

    # Root login shell, which when ended will end the docker session.
    # For batch use, remove this line.
    su -

else

    # Add a user with UID of person running docker (in ${RUN_UID})
    # Make sure the home directory is owned by the new user
    # Make sure the user can run sudo (which may or may not be installed)
    if [ -d /home/dockuser ] ; then
      chown ${RUN_UID} /home/dockuser
    fi
    adduser --disabled-password --gecos 'Docker-User' -u ${RUN_UID} dockuser
    adduser dockuser sudo

    # If root access is needed in the docker image while running as a normal
    # user, uncomment this and add 'sudo' as a packaged installed in the dockerfile
    # echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

    # We will be su'ing using a non-login shell or command, and preserving
    # the environment.  This is done so that env. variables passed in with
    # "docker run --env ..." are honored.
    # Therefore, we need to reset at least HOME=/root ...
    export HOME=/home/dockuser

    if [ -z "$RUN_CMD" ] ; then  # Launch a shell as dockuser
      su -m dockuser -c /bin/bash
    else                         # Run command as dockuser
      su -m dockuser -c "${RUN_CMD}"
    fi

fi
