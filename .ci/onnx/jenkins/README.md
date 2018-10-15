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
