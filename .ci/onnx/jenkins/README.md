# MANUAL REPRODUCTION INSTRUCTION
From directory containing CI scripts execute runCI.sh bash script:

```
cd <path-to-repo>/.ci/jenkins/
./runCI.sh
```

To remove all items created during script execution (files, directories, docker images and containers), run:

```
./runCI.sh --cleanup
```

After first run, executing the script will rerun tox tests. To rebuild nGraph and run tests use:

```
./runCI.sh --rebuild
```
