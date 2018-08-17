#  nGraph Library Sparse Checkout 
## Express Checkout Line


This repo is the lightweight version of the nGraph Library repo.

Cloning it will clone only the latest source files without the 
third-party contrib files for docker, tests, or the documentation 
that is part of the larger nGraph Library.  


    $ git clone git@github.com:NervanaSystems/ng-lite-private.git
    $ git remote set-url origin https://github.com/NervanaSystems/ngraph.git
    $ echo "sparse-checkout.git" >> .git/info/sparse-checkout
    $ git fetch
    $ git pull origin master 





